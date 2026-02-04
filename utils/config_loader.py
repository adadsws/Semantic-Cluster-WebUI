"""
Configuration Loader for Semantic-Cluster-WebUI
📅 Last Updated: 2026-01-31
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig


class ConfigLoader:
    """
    配置加载器 - 负责加载和管理config.yaml和prompts.yaml
    """
    
    def __init__(self, config_path: Optional[str] = None, prompts_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: config.yaml的路径，默认为config/config.yaml
            prompts_path: prompts.yaml的路径，默认为config/prompts.yaml
        """
        # 默认路径
        self.project_root = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else self.project_root / "config" / "config.yaml"
        self.prompts_path = Path(prompts_path) if prompts_path else self.project_root / "config" / "prompts.yaml"
        
        # 加载配置
        self.config: DictConfig = self._load_config()
        self.prompts: Dict[str, Any] = self._load_prompts()
    
    def _load_config(self) -> DictConfig:
        """
        加载config.yaml并返回OmegaConf对象
        
        Returns:
            DictConfig: 配置对象
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # 使用OmegaConf加载YAML
        config = OmegaConf.load(self.config_path)
        
        # 验证必需的顶层键
        required_keys = ['data', 'clustering', 'vlm', 'embedding', 'postprocessing', 'output', 'optimization']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config section: {key}")
        
        return config
    
    def _load_prompts(self) -> Dict[str, Any]:
        """
        加载prompts.yaml
        
        Returns:
            Dict: Prompt模板字典
        """
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")
        
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        return prompts
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的嵌套键
        
        Args:
            key: 配置键，例如 "data.input_directory" 或 "clustering.epsilon"
            default: 默认值
            
        Returns:
            配置值
            
        Example:
            >>> loader = ConfigLoader()
            >>> loader.get("clustering.epsilon")
            0.15
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，例如 "data.input_directory"
            value: 新值
            
        Example:
            >>> loader = ConfigLoader()
            >>> loader.set("clustering.epsilon", 0.2)
        """
        OmegaConf.update(self.config, key, value)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        保存当前配置到文件
        
        Args:
            output_path: 输出路径，默认覆盖原配置文件
        """
        save_path = Path(output_path) if output_path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config=self.config, f=f)
    
    def get_prompt(self, prompt_type: str, template_name: str = "default") -> str:
        """
        获取Prompt模板
        
        Args:
            prompt_type: Prompt类型，例如 "caption_prompts", "label_prompts"
            template_name: 模板名称，默认为"default"
            
        Returns:
            Prompt模板字符串
            
        Example:
            >>> loader = ConfigLoader()
            >>> loader.get_prompt("caption_prompts", "default")
        """
        if prompt_type not in self.prompts:
            raise KeyError(f"Prompt type not found: {prompt_type}")
        
        templates = self.prompts[prompt_type]
        if template_name not in templates:
            raise KeyError(f"Template '{template_name}' not found in '{prompt_type}'")
        
        return templates[template_name]
    
    def format_prompt(self, prompt_type: str, template_name: str = "default", **kwargs) -> str:
        """
        获取并格式化Prompt模板
        
        Args:
            prompt_type: Prompt类型
            template_name: 模板名称
            **kwargs: 格式化参数
            
        Returns:
            格式化后的Prompt字符串
            
        Example:
            >>> loader = ConfigLoader()
            >>> loader.format_prompt("caption_prompts", "default", caption_length=50)
        """
        template = self.get_prompt(prompt_type, template_name)
        return template.format(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为普通字典
        
        Returns:
            Dict: 配置字典
        """
        return OmegaConf.to_container(self.config, resolve=True)
    
    def __repr__(self) -> str:
        return f"ConfigLoader(config={self.config_path}, prompts={self.prompts_path})"


# ============================================
# Utility Functions
# ============================================

def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    便捷函数：加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigLoader: 配置加载器实例
    """
    return ConfigLoader(config_path=config_path)


def validate_clustering_mode(config: DictConfig) -> None:
    """
    验证聚类模式配置的一致性
    
    Args:
        config: 配置对象
        
    Raises:
        ValueError: 如果配置不一致
    """
    mode = config.clustering.mode
    
    if mode not in ["fixed_eps", "noise_control"]:
        raise ValueError(f"Invalid clustering mode: {mode}. Must be 'fixed_eps' or 'noise_control'")
    
    if mode == "fixed_eps" and config.clustering.epsilon <= 0:
        raise ValueError(f"Epsilon must be > 0 in fixed_eps mode, got {config.clustering.epsilon}")
    
    if mode == "noise_control":
        ratio = config.clustering.max_noise_ratio
        if not (0 <= ratio <= 100):
            raise ValueError(f"Max noise ratio must be 0-100%, got {ratio}")


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    # 测试配置加载器
    print("Testing ConfigLoader...")
    
    try:
        loader = ConfigLoader()
        print(f"✅ Config loaded successfully")
        print(f"   Config path: {loader.config_path}")
        print(f"   Prompts path: {loader.prompts_path}")
        
        # 测试配置访问
        print(f"\n📋 Sample Config Values:")
        print(f"   Clustering epsilon: {loader.get('clustering.epsilon')}")
        print(f"   Embedding provider: {loader.get('embedding.provider')}")
        print(f"   Batch size: {loader.get('embedding.batch_size')}")
        
        # 测试Prompt访问
        print(f"\n📝 Sample Prompt:")
        prompt = loader.format_prompt(
            "caption_prompts", 
            "default", 
            caption_length=50
        )
        print(f"   {prompt[:100]}...")
        
        # 验证聚类模式
        validate_clustering_mode(loader.config)
        print(f"\n✅ Clustering mode validation passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
