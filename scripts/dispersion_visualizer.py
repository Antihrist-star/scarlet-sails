"""
DISPERSION VISUALIZATION
Creates comprehensive charts to visualize strategy differences

Charts:
1. Distribution plots (histograms + KDE)
2. Box plots (comparing strategies)
3. Time series (P_j(S) over time)
4. Correlation heatmap
5. Scatter plot matrix
6. Violin plots

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DispersionVisualizer:
    """
    Visualization tools for dispersion analysis
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.fig_size = self.config.get('fig_size', (12, 8))
        
        logger.info("DispersionVisualizer initialized")
    
    def plot_distributions(self, vectors: Dict[str, np.ndarray], 
                          save_path: str = None) -> plt.Figure:
        """
        Plot distribution of P_j values for each strategy
        
        Parameters:
        -----------
        vectors : dict
            {strategy_name: P_j vector}
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        Figure
        """
        logger.info("Creating distribution plots...")
        
        fig, axes = plt.subplots(len(vectors), 1, figsize=(12, 4*len(vectors)))
        
        if len(vectors) == 1:
            axes = [axes]
        
        for idx, (name, vec) in enumerate(vectors.items()):
            ax = axes[idx]
            
            # Remove NaN
            clean_vec = vec[~np.isnan(vec)]
            
            if len(clean_vec) == 0:
                ax.text(0.5, 0.5, f'No data for {name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Histogram
            ax.hist(clean_vec, bins=50, alpha=0.6, density=True, 
                   label=f'{name} (n={len(clean_vec)})')
            
            # KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(clean_vec)
                x_range = np.linspace(clean_vec.min(), clean_vec.max(), 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass
            
            # Statistics
            mean_val = np.mean(clean_vec)
            median_val = np.median(clean_vec)
            
            ax.axvline(mean_val, color='blue', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='green', linestyle='--', 
                      linewidth=2, label=f'Median: {median_val:.4f}')
            
            ax.set_xlabel(f'P_{name}(S)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'Distribution of {name.upper()} Decisions', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to: {save_path}")
        
        return fig
    
    def plot_boxplots(self, vectors: Dict[str, np.ndarray],
                     save_path: str = None) -> plt.Figure:
        """
        Box plots comparing all strategies
        
        Parameters:
        -----------
        vectors : dict
            {strategy_name: P_j vector}
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        Figure
        """
        logger.info("Creating box plots...")
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Prepare data
        data = []
        labels = []
        
        for name, vec in vectors.items():
            clean_vec = vec[~np.isnan(vec)]
            if len(clean_vec) > 0:
                data.append(clean_vec)
                labels.append(name)
        
        # Box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('P_j(S) Value', fontsize=12)
        ax.set_title('Strategy Decision Values - Box Plot Comparison', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to: {save_path}")
        
        return fig
    
    def plot_time_series(self, vectors: Dict[str, np.ndarray],
                        timestamps: pd.DatetimeIndex = None,
                        save_path: str = None) -> plt.Figure:
        """
        Plot P_j(S) values over time for all strategies
        
        Parameters:
        -----------
        vectors : dict
            {strategy_name: P_j vector}
        timestamps : DatetimeIndex, optional
            Time index for x-axis
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        Figure
        """
        logger.info("Creating time series plots...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for name, vec in vectors.items():
            if timestamps is not None:
                # Align with timestamps
                valid = ~np.isnan(vec)
                ax.plot(timestamps[valid], vec[valid], label=name, alpha=0.7, linewidth=1.5)
            else:
                # Just use index
                valid = ~np.isnan(vec)
                ax.plot(np.arange(len(vec))[valid], vec[valid], label=name, alpha=0.7, linewidth=1.5)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
        ax.axhline(0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Signal threshold')
        
        ax.set_xlabel('Time' if timestamps is not None else 'Bar Index', fontsize=12)
        ax.set_ylabel('P_j(S) Value', fontsize=12)
        ax.set_title('Strategy Decisions Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to: {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame,
                                save_path: str = None) -> plt.Figure:
        """
        Heatmap of correlation matrix
        
        Parameters:
        -----------
        corr_matrix : DataFrame
            Correlation matrix
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        Figure
        """
        logger.info("Creating correlation heatmap...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Strategy Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to: {save_path}")
        
        return fig
    
    def plot_scatter_matrix(self, vectors: Dict[str, np.ndarray],
                           save_path: str = None) -> plt.Figure:
        """
        Scatter plot matrix (pairwise comparisons)
        
        Parameters:
        -----------
        vectors : dict
            {strategy_name: P_j vector}
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        Figure
        """
        logger.info("Creating scatter matrix...")
        
        # Align vectors
        aligned = self._align_vectors(vectors)
        df = pd.DataFrame(aligned)
        
        # Create scatter matrix
        from pandas.plotting import scatter_matrix
        
        fig, axes = plt.subplots(len(df.columns), len(df.columns), 
                                figsize=(12, 12))
        
        scatter_matrix(df, alpha=0.3, figsize=(12, 12), diagonal='kde', ax=axes)
        
        fig.suptitle('Strategy Scatter Matrix', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to: {save_path}")
        
        return fig
    
    def plot_violin(self, vectors: Dict[str, np.ndarray],
                   save_path: str = None) -> plt.Figure:
        """
        Violin plots showing distribution shapes
        
        Parameters:
        -----------
        vectors : dict
            {strategy_name: P_j vector}
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        Figure
        """
        logger.info("Creating violin plots...")
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Prepare data
        data = []
        labels = []
        
        for name, vec in vectors.items():
            clean_vec = vec[~np.isnan(vec)]
            if len(clean_vec) > 0:
                for val in clean_vec:
                    data.append({'Strategy': name, 'P_j(S)': val})
        
        df = pd.DataFrame(data)
        
        # Violin plot
        sns.violinplot(data=df, x='Strategy', y='P_j(S)', ax=ax, inner='box')
        
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Strategy Decision Distributions - Violin Plot', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to: {save_path}")
        
        return fig
    
    def create_full_report(self, results: Dict, output_dir: str = '.'):
        """
        Create full visualization report
        
        Parameters:
        -----------
        results : dict
            Results from DispersionAnalyzer
        output_dir : str
            Directory to save figures
        """
        logger.info("Creating full visualization report...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        vectors = results['vectors']
        
        # 1. Distributions
        self.plot_distributions(vectors, 
            save_path=os.path.join(output_dir, '1_distributions.png'))
        
        # 2. Box plots
        self.plot_boxplots(vectors,
            save_path=os.path.join(output_dir, '2_boxplots.png'))
        
        # 3. Time series
        self.plot_time_series(vectors,
            save_path=os.path.join(output_dir, '3_timeseries.png'))
        
        # 4. Correlation heatmap
        if 'correlation_matrix' in results:
            self.plot_correlation_heatmap(results['correlation_matrix'],
                save_path=os.path.join(output_dir, '4_correlation.png'))
        
        # 5. Scatter matrix
        self.plot_scatter_matrix(vectors,
            save_path=os.path.join(output_dir, '5_scatter_matrix.png'))
        
        # 6. Violin plots
        self.plot_violin(vectors,
            save_path=os.path.join(output_dir, '6_violin.png'))
        
        logger.info(f"✅ Full report created in: {output_dir}")
        logger.info("   Files: 1-6 .png")
        
        plt.close('all')
    
    def _align_vectors(self, vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align vectors by removing NaN"""
        all_valid = None
        
        for vec in vectors.values():
            valid = ~np.isnan(vec)
            if all_valid is None:
                all_valid = valid
            else:
                all_valid = all_valid & valid
        
        aligned = {}
        for name, vec in vectors.items():
            aligned[name] = vec[all_valid]
        
        return aligned


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("DISPERSION VISUALIZER TEST")
    print("="*80)
    
    print("\nModule loaded successfully!")
    print("Use: visualizer = DispersionVisualizer()")
    print("Then: visualizer.create_full_report(results)")