"""
File Utilities for Time Series Prediction Agent
时序预测代理系统的文件工具模块
"""

import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import shutil
import zipfile
import tempfile

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathManager:
    """路径管理器"""
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            "logs",
            "cache", 
            "models",
            "visualizations",
            "reports",
            "data",
            "temp"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def get_path(self, category: str, filename: str = None) -> Path:
        """获取指定类别的路径"""
        category_path = self.base_dir / category
        
        if filename:
            return category_path / filename
        return category_path
    
    def get_log_path(self, filename: str = None) -> Path:
        """获取日志路径"""
        return self.get_path("logs", filename)
    
    def get_cache_path(self, filename: str = None) -> Path:
        """获取缓存路径"""
        return self.get_path("cache", filename)
    
    def get_model_path(self, filename: str = None) -> Path:
        """获取模型路径"""
        return self.get_path("models", filename)
    
    def get_visualization_path(self, filename: str = None) -> Path:
        """获取可视化路径"""
        return self.get_path("visualizations", filename)
    
    def get_report_path(self, filename: str = None) -> Path:
        """获取报告路径"""
        return self.get_path("reports", filename)
    
    def get_data_path(self, filename: str = None) -> Path:
        """获取数据路径"""
        return self.get_path("data", filename)
    
    def get_temp_path(self, filename: str = None) -> Path:
        """获取临时文件路径"""
        return self.get_path("temp", filename)
    
    def create_experiment_dir(self, experiment_name: str = None) -> Path:
        """创建实验目录"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        experiment_dir = self.base_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 在实验目录下创建子目录
        for subdir in ["logs", "models", "visualizations", "reports", "data"]:
            (experiment_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Created experiment directory: {experiment_dir}")
        return experiment_dir
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """清理临时文件"""
        temp_dir = self.get_temp_path()
        current_time = datetime.now()
        
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    file_path.unlink()
                    logger.debug(f"Cleaned up temp file: {file_path}")


class FileSaver:
    """文件保存器"""
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Union[str, Path], 
                  indent: int = 2, ensure_ascii: bool = False, **kwargs):
        """Save JSON file, handling numpy and pandas types"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        import numpy as np
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, (pd.api.extensions.ExtensionDtype, np.dtype)):
                return str(obj)
            return obj
        try:
            # Recursively convert numpy/pandas types
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            converted_data = recursive_convert(data)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
            # logger.info(f"Successfully saved JSON file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON file {filepath}: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path], **kwargs):
        """保存Pickle文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, **kwargs)
            logger.info(f"Successfully saved pickle file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pickle file {filepath}: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: Union[str, Path], **kwargs):
        """保存YAML文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, **kwargs)
            logger.info(f"Successfully saved YAML file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save YAML file {filepath}: {e}")
            raise
    
    @staticmethod
    def save_csv(data: Any, filepath: Union[str, Path], **kwargs):
        """保存CSV文件"""
        import pandas as pd
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, **kwargs)
            else:
                pd.DataFrame(data).to_csv(filepath, **kwargs)
            logger.info(f"Successfully saved CSV file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CSV file {filepath}: {e}")
            raise
    
    @staticmethod
    def save_text(text: str, filepath: Union[str, Path], encoding: str = 'utf-8', **kwargs):
        """保存文本文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(text)
            logger.info(f"Successfully saved text file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save text file {filepath}: {e}")
            raise


class FileLoader:
    """文件加载器"""
    
    @staticmethod
    def load_json(filepath: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """加载JSON文件"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f, **kwargs)
            logger.info(f"Successfully loaded JSON file: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_pickle(filepath: Union[str, Path], **kwargs) -> Any:
        """加载Pickle文件"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f, **kwargs)
            logger.info(f"Successfully loaded pickle file: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """加载YAML文件"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"YAML file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f, **kwargs)
            logger.info(f"Successfully loaded YAML file: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load YAML file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_csv(filepath: Union[str, Path], **kwargs):
        """加载CSV文件"""
        import pandas as pd
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            data = pd.read_csv(filepath, **kwargs)
            logger.info(f"Successfully loaded CSV file: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load CSV file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_text(filepath: Union[str, Path], encoding: str = 'utf-8', **kwargs) -> str:
        """加载文本文件"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Text file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            logger.info(f"Successfully loaded text file: {filepath}")
            return text
        except Exception as e:
            logger.error(f"Failed to load text file {filepath}: {e}")
            raise


class FileManager:
    """文件管理器"""
    
    def __init__(self, base_dir: str = "results"):
        self.path_manager = PathManager(base_dir)
        self.saver = FileSaver()
        self.loader = FileLoader()
    
    def save_experiment_results(self, 
                               experiment_name: str,
                               results: Dict[str, Any],
                               save_individual: bool = True) -> Dict[str, Path]:
        """保存实验结果"""
        experiment_dir = self.path_manager.create_experiment_dir(experiment_name)
        saved_files = {}
        
        # 保存综合结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式的综合报告
        report_path = experiment_dir / "reports" / f"comprehensive_report_{timestamp}.json"
        self.saver.save_json(results, report_path)
        saved_files['comprehensive_report'] = report_path
        
        # 保存YAML格式的配置
        if 'config' in results:
            config_path = experiment_dir / f"config_{timestamp}.yaml"
            self.saver.save_yaml(results['config'], config_path)
            saved_files['config'] = config_path
        
        # 保存预测结果
        if 'forecasts' in results:
            forecast_path = experiment_dir / "data" / f"forecasts_{timestamp}.json"
            self.saver.save_json(results['forecasts'], forecast_path)
            saved_files['forecasts'] = forecast_path
        
        # 保存指标
        if 'metrics' in results:
            metrics_path = experiment_dir / "data" / f"metrics_{timestamp}.json"
            self.saver.save_json(results['metrics'], metrics_path)
            saved_files['metrics'] = metrics_path
        
        # 保存可视化路径
        if 'visualizations' in results:
            viz_path = experiment_dir / "data" / f"visualizations_{timestamp}.json"
            self.saver.save_json(results['visualizations'], viz_path)
            saved_files['visualizations'] = viz_path
        
        # 保存实验摘要
        summary = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'saved_files': {k: str(v) for k, v in saved_files.items()},
            'results_summary': {
                'num_slices': len(results.get('slice_results', [])),
                'selected_models': results.get('selected_models', []),
                'best_metrics': results.get('best_metrics', {})
            }
        }
        
        summary_path = experiment_dir / f"experiment_summary_{timestamp}.json"
        self.saver.save_json(summary, summary_path)
        saved_files['summary'] = summary_path
        
        logger.info(f"Saved experiment results to: {experiment_dir}")
        return saved_files
    
    def load_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """加载实验结果"""
        experiment_dir = self.path_manager.base_dir / experiment_name
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        results = {}
        
        # 加载综合报告
        report_files = list(experiment_dir.glob("reports/comprehensive_report_*.json"))
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            results['comprehensive_report'] = self.loader.load_json(latest_report)
        
        # 加载配置
        config_files = list(experiment_dir.glob("config_*.yaml"))
        if config_files:
            latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
            results['config'] = self.loader.load_yaml(latest_config)
        
        # 加载预测结果
        forecast_files = list(experiment_dir.glob("data/forecasts_*.json"))
        if forecast_files:
            latest_forecast = max(forecast_files, key=lambda x: x.stat().st_mtime)
            results['forecasts'] = self.loader.load_json(latest_forecast)
        
        # 加载指标
        metrics_files = list(experiment_dir.glob("data/metrics_*.json"))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
            results['metrics'] = self.loader.load_json(latest_metrics)
        
        return results
    
    def create_backup(self, source_dir: str, backup_name: str = None) -> Path:
        """创建备份"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")
        
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.path_manager.base_dir / "backups" / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 创建ZIP备份
        zip_path = backup_path.with_suffix('.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created backup: {zip_path}")
        return zip_path
    
    def cleanup_old_experiments(self, max_age_days: int = 30):
        """清理旧的实验目录"""
        current_time = datetime.now()
        
        for experiment_dir in self.path_manager.base_dir.iterdir():
            if experiment_dir.is_dir() and experiment_dir.name.startswith('experiment_'):
                dir_age = current_time - datetime.fromtimestamp(experiment_dir.stat().st_mtime)
                if dir_age.days > max_age_days:
                    shutil.rmtree(experiment_dir)
                    logger.info(f"Cleaned up old experiment: {experiment_dir}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], name: str):
        """保存配置"""
        config_path = self.config_dir / f"{name}.yaml"
        FileSaver.save_yaml(config, config_path)
        logger.info(f"Saved config: {config_path}")
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """加载配置"""
        config_path = self.config_dir / f"{name}.yaml"
        return FileLoader.load_yaml(config_path)
    
    def list_configs(self) -> List[str]:
        """列出所有配置"""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]
    
    def delete_config(self, name: str):
        """删除配置"""
        config_path = self.config_dir / f"{name}.yaml"
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Deleted config: {config_path}")


def create_temp_file(suffix: str = "", prefix: str = "temp_", dir: str = None) -> Path:
    """创建临时文件"""
    if dir is None:
        dir = Path.cwd() / "temp"
        dir.mkdir(exist_ok=True)
    
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix, 
        prefix=prefix, 
        dir=dir, 
        delete=False
    )
    temp_file.close()
    
    return Path(temp_file.name)


def get_file_size(filepath: Union[str, Path]) -> int:
    """获取文件大小（字节）"""
    filepath = Path(filepath)
    if filepath.exists():
        return filepath.stat().st_size
    return 0


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """获取文件信息"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {"exists": False}
    
    stat = filepath.stat()
    
    return {
        "exists": True,
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
        "is_file": filepath.is_file(),
        "is_dir": filepath.is_dir(),
        "extension": filepath.suffix,
        "name": filepath.name,
        "stem": filepath.stem,
        "parent": str(filepath.parent)
    } 