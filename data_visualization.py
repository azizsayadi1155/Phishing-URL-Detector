import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataVisualizer:
    def __init__(self, data_path='data/phishing_dataset.csv'):
        """Initialize the visualizer with data path"""
        self.df = pd.read_csv(data_path)
        self.create_output_dir()
    
    def create_output_dir(self):
        """Create directory for saving visualizations"""
        import os
        os.makedirs('visualizations', exist_ok=True)
    
    def plot_class_distribution(self):
        """Plot the distribution of phishing vs legitimate URLs"""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='label')
        plt.title('Distribution of Phishing vs Legitimate URLs')
        plt.xlabel('Label (0: Legitimate, 1: Phishing)')
        plt.ylabel('Count')
        plt.savefig('visualizations/class_distribution.png')
        plt.close()
        
        # Calculate percentages
        total = len(self.df)
        percentages = self.df['label'].value_counts(normalize=True) * 100
        
        # Create pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(percentages, labels=['Phishing', 'Legitimate'], 
                autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        plt.title('Percentage Distribution of URLs')
        plt.savefig('visualizations/class_distribution_pie.png')
        plt.close()
    
    def plot_url_length_distribution(self):
        """Plot URL length distribution for both classes"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='label', y='URLLength')
        plt.title('URL Length Distribution by Class')
        plt.xlabel('Label (0: Legitimate, 1: Phishing)')
        plt.ylabel('URL Length')
        plt.savefig('visualizations/url_length_distribution.png')
        plt.close()
        
        # Kernel Density Estimation plot
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=self.df[self.df['label']==0]['URLLength'], 
                   label='Legitimate', color='blue')
        sns.kdeplot(data=self.df[self.df['label']==1]['URLLength'], 
                   label='Phishing', color='red')
        plt.title('URL Length Density Distribution')
        plt.xlabel('URL Length')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('visualizations/url_length_density.png')
        plt.close()
    
    def plot_feature_correlations(self):
        """Plot correlation matrix of numerical features"""
        # Select numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/feature_correlations.png')
        plt.close()
    
    def plot_top_tlds(self):
        """Plot distribution of top TLDs"""
        plt.figure(figsize=(12, 6))
        self.df['TLD'].value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Most Common TLDs')
        plt.xlabel('TLD')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/top_tlds.png')
        plt.close()
    
    def plot_security_features(self):
        """Plot distribution of security-related features"""
        security_features = ['IsHTTPS', 'HasFavicon', 'HasTitle', 
                           'HasPasswordField', 'HasHiddenFields']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(security_features):
            sns.countplot(data=self.df, x=feature, hue='label', ax=axes[idx])
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].legend(['Legitimate', 'Phishing'])
        
        plt.tight_layout()
        plt.savefig('visualizations/security_features.png')
        plt.close()
    
    def plot_special_chars_analysis(self):
        """Plot analysis of special characters in URLs"""
        special_chars = ['NoOfEqualsInURL', 'NoOfQMarkInURL', 
                        'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL']
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=special_chars)
        
        for idx, feature in enumerate(special_chars):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            # Create violin plot
            fig.add_trace(
                go.Violin(x=self.df['label'].map({0: 'Legitimate', 1: 'Phishing'}),
                         y=self.df[feature],
                         name=feature,
                         box_visible=True,
                         meanline_visible=True),
                row=row, col=col
            )
        
        fig.update_layout(height=800, width=1000, 
                         title_text="Special Characters Analysis")
        fig.write_html('visualizations/special_chars_analysis.html')
    
    def plot_domain_analysis(self):
        """Plot domain-related features analysis"""
        domain_features = ['DomainLength', 'NoOfSubDomain', 'TLDLength']
        
        fig = plt.figure(figsize=(15, 5))
        for idx, feature in enumerate(domain_features, 1):
            plt.subplot(1, 3, idx)
            sns.boxplot(data=self.df, x='label', y=feature)
            plt.title(f'{feature} by Class')
            plt.xlabel('Label (0: Legitimate, 1: Phishing)')
        
        plt.tight_layout()
        plt.savefig('visualizations/domain_analysis.png')
        plt.close()
    
    def create_interactive_scatter(self):
        """Create interactive scatter plot of key features"""
        fig = px.scatter(self.df, x='URLLength', y='NoOfSubDomain',
                        color='label', size='DomainLength',
                        hover_data=['TLD', 'IsHTTPS'],
                        title='URL Characteristics Interactive Plot',
                        labels={'label': 'URL Type'},
                        color_discrete_map={0: 'blue', 1: 'red'})
        
        fig.write_html('visualizations/interactive_scatter.html')
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        self.plot_class_distribution()
        print("✓ Class distribution plots generated")
        
        self.plot_url_length_distribution()
        print("✓ URL length distribution plots generated")
        
        self.plot_feature_correlations()
        print("✓ Feature correlation matrix generated")
        
        self.plot_top_tlds()
        print("✓ Top TLDs plot generated")
        
        self.plot_security_features()
        print("✓ Security features plots generated")
        
        self.plot_special_chars_analysis()
        print("✓ Special characters analysis generated")
        
        self.plot_domain_analysis()
        print("✓ Domain analysis plots generated")
        
        self.create_interactive_scatter()
        print("✓ Interactive scatter plot generated")
        
        print("\nAll visualizations have been saved to the 'visualizations' directory")

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.generate_all_visualizations()
