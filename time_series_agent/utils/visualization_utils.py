"""
Visualization Utilities for Time Series Prediction Agent
时序预测代理系统的可视化工具模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TimeSeriesVisualizer:
    """时序数据可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.figure_size = config.get('visualization', {}).get('figure_size', (12, 8))
        self.dpi = config.get('visualization', {}).get('dpi', 300)
        self.save_format = config.get('visualization', {}).get('save_format', 'png')
        self.show_plots = config.get('visualization', {}).get('show_plots', False)
    
    def plot_time_series(self, data: pd.DataFrame, 
                        title: str = "Time Series Data",
                        save_path: Optional[str] = None) -> str:
        """绘制时间序列图"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 绘制时间序列
        ax.plot(data.index, data['value'], linewidth=1, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        
        plt.close()
        return save_path or ""
    
    def plot_data_quality(self, data: pd.DataFrame,
                         missing_info: Dict[str, Any],
                         outlier_info: Dict[str, Any],
                         save_path: Optional[str] = None) -> str:
        """绘制数据质量分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 时间序列图
        axes[0, 0].plot(data.index, data['value'], linewidth=1)
        axes[0, 0].set_title('Time Series Data')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 缺失值分布
        missing_data = data.isnull()
        if missing_data.any().any():
            missing_data.sum().plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values by Column')
            axes[0, 1].set_ylabel('Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Missing Values')
        
        # 3. 异常值检测
        if outlier_info and 'value' in outlier_info:
            outlier_indices = outlier_info['value']
            if outlier_indices:
                axes[1, 0].plot(data.index, data['value'], linewidth=1, alpha=0.7)
                axes[1, 0].scatter(data.index[outlier_indices], 
                                 data['value'].iloc[outlier_indices], 
                                 color='red', s=20, alpha=0.8, label='Outliers')
                axes[1, 0].set_title('Outlier Detection')
                axes[1, 0].legend()
            else:
                axes[1, 0].plot(data.index, data['value'], linewidth=1)
                axes[1, 0].set_title('No Outliers Detected')
        else:
            axes[1, 0].plot(data.index, data['value'], linewidth=1)
            axes[1, 0].set_title('Outlier Detection')
        
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 数据分布
        axes[1, 1].hist(data['value'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Data Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Data quality plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        
        plt.close()
        return save_path or ""
    
    def plot_analysis_results(self, data: pd.DataFrame,
                            analysis: Dict[str, Any],
                            save_path: Optional[str] = None) -> str:
        """绘制分析结果图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 原始时间序列
        axes[0, 0].plot(data.index, data['value'], linewidth=1)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 趋势分析
        trend_analysis = analysis.get('trend_analysis', {})
        if trend_analysis.get('has_trend'):
            slope = trend_analysis.get('slope', 0)
            intercept = trend_analysis.get('intercept', 0)
            x = np.arange(len(data))
            trend_line = slope * x + intercept
            axes[0, 1].plot(data.index, data['value'], linewidth=1, alpha=0.7, label='Data')
            axes[0, 1].plot(data.index, trend_line, 'r--', linewidth=2, label=f'Trend (slope={slope:.4f})')
            axes[0, 1].set_title(f'Trend Analysis - {trend_analysis.get("trend_direction", "unknown")}')
            axes[0, 1].legend()
        else:
            axes[0, 1].plot(data.index, data['value'], linewidth=1)
            axes[0, 1].set_title('No Significant Trend Detected')
        
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 季节性分析
        seasonality_analysis = analysis.get('seasonality_analysis', {})
        if seasonality_analysis.get('has_seasonality'):
            seasonal_period = seasonality_analysis.get('seasonal_period')
            axes[1, 0].plot(data.index, data['value'], linewidth=1)
            axes[1, 0].set_title(f'Seasonality Detected (Period: {seasonal_period})')
        else:
            axes[1, 0].plot(data.index, data['value'], linewidth=1)
            axes[1, 0].set_title('No Significant Seasonality')
        
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 统计信息
        basic_stats = analysis.get('basic_stats', {}).get('value', {})
        if basic_stats:
            stats_text = f"""
            Mean: {basic_stats.get('mean', 'N/A'):.2f}
            Std: {basic_stats.get('std', 'N/A'):.2f}
            Min: {basic_stats.get('min', 'N/A'):.2f}
            Max: {basic_stats.get('max', 'N/A'):.2f}
            Skewness: {basic_stats.get('skewness', 'N/A'):.2f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Statistical Summary')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Analysis results plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        
        plt.close()
        return save_path or ""
    
    def plot_forecast_comparison(self, 
                               actual: pd.Series,
                               predictions: Dict[str, List[float]],
                               ensemble_pred: List[float],
                               save_path: Optional[str] = None) -> str:
        """绘制预测对比图"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. 预测对比图
        axes[0].plot(actual.index, actual.values, 'k-', linewidth=2, label='Actual', alpha=0.8)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(predictions)))
        for i, (model, pred) in enumerate(predictions.items()):
            if len(pred) == len(actual):
                axes[0].plot(actual.index, pred, '--', linewidth=1, 
                           label=model, color=colors[i], alpha=0.7)
        
        if len(ensemble_pred) == len(actual):
            axes[0].plot(actual.index, ensemble_pred, 'r-', linewidth=2, 
                        label='Ensemble', alpha=0.9)
        
        axes[0].set_title('Forecast Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. 预测误差图
        if len(ensemble_pred) == len(actual):
            errors = np.array(ensemble_pred) - actual.values
            axes[1].plot(actual.index, errors, 'r-', linewidth=1, alpha=0.7)
            axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1].set_title('Forecast Errors (Ensemble)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Error')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Forecast comparison plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        
        plt.close()
        return save_path or ""
    
    def plot_model_performance(self, metrics: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None) -> str:
        """绘制模型性能对比图"""
        if not metrics:
            return ""
        
        # 准备数据
        models = list(metrics.keys())
        mse_values = [metrics[model].get('mse', 0) for model in models]
        mae_values = [metrics[model].get('mae', 0) for model in models]
        mape_values = [metrics[model].get('mape', 0) for model in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MSE对比
        axes[0].bar(models, mse_values, alpha=0.7)
        axes[0].set_title('MSE Comparison', fontweight='bold')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE对比
        axes[1].bar(models, mae_values, alpha=0.7)
        axes[1].set_title('MAE Comparison', fontweight='bold')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAPE对比
        axes[2].bar(models, mape_values, alpha=0.7)
        axes[2].set_title('MAPE Comparison', fontweight='bold')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        
        plt.close()
        return save_path or ""
    
    def plot_ensemble_forecast_with_confidence(self,
                                             data: pd.DataFrame,
                                             individual_forecasts: Dict[str, List[float]],
                                             ensemble_forecast: List[float],
                                             confidence_level: float = 0.95,
                                             save_path: Optional[str] = None) -> str:
        """绘制带置信区间的集成预测图"""
        if not individual_forecasts:
            return ""
        
        # 计算置信区间
        forecasts_array = np.array(list(individual_forecasts.values()))
        ensemble_mean = np.mean(forecasts_array, axis=0)
        ensemble_std = np.std(forecasts_array, axis=0)
        
        # 计算置信区间
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        confidence_interval = z_score * ensemble_std
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 绘制历史数据
        ax.plot(data.index, data['value'], 'k-', linewidth=2, label='Historical Data', alpha=0.8)
        
        # 创建预测时间索引
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=len(ensemble_forecast), freq='D')
        else:
            forecast_dates = range(len(data), len(data) + len(ensemble_forecast))
        
        # 绘制预测
        ax.plot(forecast_dates, ensemble_forecast, 'r-', linewidth=2, label='Ensemble Forecast')
        
        # 绘制置信区间
        ax.fill_between(forecast_dates, 
                       ensemble_forecast - confidence_interval,
                       ensemble_forecast + confidence_interval,
                       alpha=0.3, color='red', label=f'{confidence_level*100:.0f}% Confidence Interval')
        
        # 绘制各模型预测
        colors = plt.cm.Set3(np.linspace(0, 1, len(individual_forecasts)))
        for i, (model, pred) in enumerate(individual_forecasts.items()):
            ax.plot(forecast_dates, pred, '--', linewidth=1, 
                   label=model, color=colors[i], alpha=0.5)
        
        ax.set_title('Ensemble Forecast with Confidence Interval', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Ensemble forecast plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        
        plt.close()
        return save_path or ""
    
    def create_interactive_plot(self, data: pd.DataFrame,
                              predictions: Dict[str, List[float]],
                              ensemble_pred: List[float]) -> go.Figure:
        """创建交互式Plotly图表"""
        fig = go.Figure()
        
        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['value'],
            mode='lines',
            name='Historical Data',
            line=dict(color='black', width=2)
        ))
        
        # 创建预测时间索引
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=len(ensemble_pred), freq='D')
        else:
            forecast_dates = range(len(data), len(data) + len(ensemble_pred))
        
        # 添加各模型预测
        colors = px.colors.qualitative.Set3
        for i, (model, pred) in enumerate(predictions.items()):
            if len(pred) == len(forecast_dates):
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=pred,
                    mode='lines',
                    name=model,
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                    opacity=0.7
                ))
        
        # 添加集成预测
        if len(ensemble_pred) == len(forecast_dates):
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=ensemble_pred,
                mode='lines',
                name='Ensemble Forecast',
                line=dict(color='red', width=3)
            ))
        
        fig.update_layout(
            title='Interactive Time Series Forecast',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def save_interactive_plot(self, fig: go.Figure, save_path: str):
        """保存交互式图表"""
        fig.write_html(save_path)
        logger.info(f"Interactive plot saved to {save_path}")


class ReportVisualizer:
    """报告可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.visualizer = TimeSeriesVisualizer(config)
    
    def create_experiment_summary_plots(self, 
                                      all_results: List[Dict[str, Any]],
                                      save_dir: str) -> Dict[str, str]:
        """创建实验总结图表"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # 1. 各切片性能对比
        slice_metrics = []
        for res in all_results:
            metrics = res.get('metrics', {}).get('ensemble', {})
            slice_metrics.append({
                'slice_id': res['slice_id'],
                'mse': metrics.get('mse', 0),
                'mae': metrics.get('mae', 0),
                'mape': metrics.get('mape', 0)
            })
        
        if slice_metrics:
            df_metrics = pd.DataFrame(slice_metrics)
            
            # MSE趋势
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_metrics['slice_id'], df_metrics['mse'], 'o-', linewidth=2, markersize=6)
            ax.set_title('MSE Across Slices', fontweight='bold')
            ax.set_xlabel('Slice ID')
            ax.set_ylabel('MSE')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            mse_plot_path = save_dir / "mse_across_slices.png"
            plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
            plots['mse_trend'] = str(mse_plot_path)
            plt.close()
            
            # MAPE趋势
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_metrics['slice_id'], df_metrics['mape'], 'o-', linewidth=2, markersize=6, color='orange')
            ax.set_title('MAPE Across Slices', fontweight='bold')
            ax.set_xlabel('Slice ID')
            ax.set_ylabel('MAPE (%)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            mape_plot_path = save_dir / "mape_across_slices.png"
            plt.savefig(mape_plot_path, dpi=300, bbox_inches='tight')
            plots['mape_trend'] = str(mape_plot_path)
            plt.close()
        
        # 2. 模型选择频率
        model_counts = {}
        for res in all_results:
            for model in res.get('selected_models', []):
                model_counts[model] = model_counts.get(model, 0) + 1
        
        if model_counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(model_counts.keys())
            counts = list(model_counts.values())
            
            bars = ax.bar(models, counts, alpha=0.7)
            ax.set_title('Model Selection Frequency', fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel('Selection Count')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            
            model_freq_path = save_dir / "model_selection_frequency.png"
            plt.savefig(model_freq_path, dpi=300, bbox_inches='tight')
            plots['model_frequency'] = str(model_freq_path)
            plt.close()
        
        return plots


def create_visualization_suite(data: pd.DataFrame,
                             analysis: Dict[str, Any],
                             predictions: Dict[str, List[float]],
                             metrics: Dict[str, Dict[str, float]],
                             config: Dict[str, Any],
                             output_dir: str) -> Dict[str, str]:
    """创建完整的可视化套件"""
    visualizer = TimeSeriesVisualizer(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # 1. 时间序列图
    ts_plot_path = output_path / "time_series.png"
    plots['time_series'] = visualizer.plot_time_series(
        data, "Time Series Data", str(ts_plot_path)
    )
    
    # 2. 分析结果图
    analysis_plot_path = output_path / "analysis_results.png"
    plots['analysis'] = visualizer.plot_analysis_results(
        data, analysis, str(analysis_plot_path)
    )
    
    # 3. 预测对比图
    if predictions and 'ensemble' in predictions:
        forecast_plot_path = output_path / "forecast_comparison.png"
        plots['forecast'] = visualizer.plot_forecast_comparison(
            data['value'], predictions, predictions['ensemble'], str(forecast_plot_path)
        )
    
    # 4. 模型性能图
    if metrics:
        performance_plot_path = output_path / "model_performance.png"
        plots['performance'] = visualizer.plot_model_performance(
            metrics, str(performance_plot_path)
        )
    
    return plots 