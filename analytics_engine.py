import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

class WrapStatAnalyticsEngine:
    """
    Advanced analytics engine for WrapStat training data
    Provides real-time ML predictions and insights
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.feature_importance = None
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("🔄 Loading and preparing data...")
        self.data = pd.read_csv(self.data_path)
        
        # Create additional derived features
        self.data['satisfaction_overall'] = self.data[['content_engaging_numeric', 'content_relevant_numeric', 
                                                      'content_understandable_numeric', 'content_visual_support_numeric']].mean(axis=1)
        
        # Training effectiveness score
        self.data['training_effectiveness'] = (
            self.data['knowledge_improvement'] * 0.5 +
            self.data['satisfaction_overall'] * 0.3 +
            (self.data['content_interactive_numeric'] / 5) * 0.2
        )
        
        # Learning velocity (improvement per minute)
        self.data['learning_velocity'] = self.data['knowledge_improvement'] / (self.data['training_duration_minutes'] + 0.1)
        
        print(f"✅ Data loaded: {len(self.data)} records with {self.data.shape[1]} features")
    
    def train_predictive_models(self):
        """Train machine learning models"""
        print("\n🤖 Training predictive models...")
        
        # Features for prediction
        feature_cols = [
            'content_engaging_numeric', 'content_relevant_numeric', 'content_understandable_numeric',
            'content_interactive_numeric', 'content_visual_support_numeric', 'training_duration_minutes',
            'knowledge_before_numeric', 'content_user_friendly_numeric', 'tech_easy_access_numeric'
        ]
        
        X = self.data[feature_cols].fillna(self.data[feature_cols].mean())
        y = self.data['knowledge_improvement']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest Model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.rf_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Random Forest Model trained:")
        print(f"   R² Score: {r2:.3f}")
        print(f"   RMSE: {rmse:.3f}")
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'feature_importance': self.feature_importance.to_dict('records')
        }
    
    def perform_clustering(self):
        """Perform participant clustering"""
        print("\n👥 Performing clustering analysis...")
        
        # Features for clustering
        cluster_features = [
            'knowledge_before_numeric', 'knowledge_improvement', 'satisfaction_overall',
            'training_duration_minutes', 'content_interactive_numeric'
        ]
        
        X_cluster = self.data[cluster_features].fillna(self.data[cluster_features].mean())
        X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
        
        # K-means clustering
        self.cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = self.cluster_model.fit_predict(X_cluster_scaled)
        
        self.data['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = []
        cluster_names = ['High Performers', 'Fast Learners', 'Struggling Learners']
        
        for i in range(3):
            cluster_data = self.data[self.data['cluster'] == i]
            
            analysis = {
                'cluster_id': i,
                'cluster_name': cluster_names[i],
                'size': len(cluster_data),
                'avg_knowledge_baseline': cluster_data['knowledge_before_numeric'].mean(),
                'avg_knowledge_improvement': cluster_data['knowledge_improvement'].mean(),
                'avg_satisfaction': cluster_data['satisfaction_overall'].mean(),
                'avg_duration': cluster_data['training_duration_minutes'].mean(),
                'characteristics': self._get_cluster_characteristics(cluster_data)
            }
            cluster_analysis.append(analysis)
        
        print(f"✅ Clustering completed: 3 segments identified")
        return cluster_analysis
    
    def _get_cluster_characteristics(self, cluster_data):
        """Get characteristics for a cluster"""
        chars = []
        
        baseline = cluster_data['knowledge_before_numeric'].mean()
        improvement = cluster_data['knowledge_improvement'].mean()
        satisfaction = cluster_data['satisfaction_overall'].mean()
        
        if baseline > 3.5:
            chars.append("High initial knowledge")
        elif baseline < 2.5:
            chars.append("Low initial knowledge")
        else:
            chars.append("Moderate initial knowledge")
            
        if improvement > 1.5:
            chars.append("High learning gains")
        elif improvement < 0.8:
            chars.append("Low learning gains")
        else:
            chars.append("Moderate learning gains")
            
        if satisfaction > 4.0:
            chars.append("High satisfaction")
        elif satisfaction < 3.5:
            chars.append("Needs improvement")
        else:
            chars.append("Moderate satisfaction")
        
        return chars
    
    def analyze_correlations(self):
        """Analyze feature correlations"""
        print("\n📊 Analyzing correlations...")
        
        correlation_features = [
            'knowledge_improvement', 'knowledge_before_numeric', 'content_engaging_numeric',
            'content_relevant_numeric', 'content_interactive_numeric', 'content_visual_support_numeric',
            'training_duration_minutes', 'satisfaction_overall'
        ]
        
        corr_matrix = self.data[correlation_features].corr()
        
        # Get top correlations with knowledge improvement
        ki_correlations = corr_matrix['knowledge_improvement'].drop('knowledge_improvement').abs().sort_values(ascending=False)
        
        correlations = {
            'matrix': corr_matrix.to_dict(),
            'top_correlations': [
                {'feature': feat, 'correlation': corr_matrix.loc[feat, 'knowledge_improvement']}
                for feat in ki_correlations.head(10).index
            ]
        }
        
        print(f"✅ Correlation analysis completed")
        return correlations
    
    def detect_anomalies(self):
        """Detect anomalous responses"""
        print("\n🚨 Detecting anomalies...")
        
        # Z-score based anomaly detection
        from scipy import stats
        
        # Check for negative knowledge change
        negative_change = self.data[self.data['knowledge_improvement'] < 0]
        
        # Statistical outliers
        z_scores = np.abs(stats.zscore(self.data[['knowledge_improvement', 'satisfaction_overall']].fillna(0)))
        outliers = self.data[(z_scores > 2).any(axis=1)]
        
        anomalies = {
            'negative_knowledge_change': {
                'count': len(negative_change),
                'responses': negative_change[['response_id', 'knowledge_before_numeric', 'knowledge_after_numeric', 'knowledge_improvement']].to_dict('records')
            },
            'statistical_outliers': {
                'count': len(outliers),
                'threshold': 2.0,
                'percentage': (len(outliers) / len(self.data)) * 100
            }
        }
        
        print(f"✅ Anomaly detection completed: {len(negative_change)} negative changes, {len(outliers)} outliers")
        return anomalies
    
    def generate_recommendations(self):
        """Generate AI-powered recommendations"""
        print("\n💡 Generating recommendations...")
        
        # Analyze improvement opportunities
        content_scores = {
            'content_interactive_numeric': self.data['content_interactive_numeric'].mean(),
            'content_engaging_numeric': self.data['content_engaging_numeric'].mean(),
            'content_understandable_numeric': self.data['content_understandable_numeric'].mean(),
            'content_visual_support_numeric': self.data['content_visual_support_numeric'].mean()
        }
        
        # Calculate improvement potential based on feature importance
        recommendations = []
        
        for _, row in self.feature_importance.head(5).iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature in content_scores:
                current_score = content_scores[feature]
                potential = (5.0 - current_score) * importance
                
                recommendations.append({
                    'feature': feature,
                    'current_score': current_score,
                    'importance': importance,
                    'improvement_potential': potential,
                    'priority': 'High' if potential > 0.15 else 'Medium' if potential > 0.08 else 'Low'
                })
        
        # Sort by improvement potential
        recommendations.sort(key=lambda x: x['improvement_potential'], reverse=True)
        
        print(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations
    
    def predict_knowledge_improvement(self, features):
        """Predict knowledge improvement for given features"""
        if self.rf_model is None:
            raise ValueError("Model not trained. Call train_predictive_models() first.")
        
        # Prepare features array
        feature_array = np.array(features).reshape(1, -1)
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.rf_model.predict(feature_array_scaled)[0]
        
        # Get prediction confidence (using feature importance)
        confidence = np.sum(self.rf_model.feature_importances_ * np.abs(feature_array[0]))
        
        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'interpretation': self._interpret_prediction(prediction)
        }
    
    def _interpret_prediction(self, prediction):
        """Interpret prediction value"""
        if prediction >= 2.0:
            return "Excellent knowledge improvement expected"
        elif prediction >= 1.5:
            return "Good knowledge improvement expected"
        elif prediction >= 1.0:
            return "Moderate knowledge improvement expected"
        elif prediction >= 0.5:
            return "Some knowledge improvement expected"
        else:
            return "Limited knowledge improvement expected"
    
    def run_comprehensive_analysis(self):
        """Run all analytics and return comprehensive results"""
        print("🚀 Running comprehensive analytics...")
        
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'total_responses': len(self.data),
                'avg_knowledge_improvement': self.data['knowledge_improvement'].mean(),
                'avg_satisfaction': self.data['satisfaction_overall'].mean(),
                'completion_rate': 100.0  # Assuming all responses are complete
            }
        }
        
        # Run all analyses
        results['predictive_model'] = self.train_predictive_models()
        results['clustering'] = self.perform_clustering()
        results['correlations'] = self.analyze_correlations()
        results['anomalies'] = self.detect_anomalies()
        results['recommendations'] = self.generate_recommendations()
        
        # Save results
        with open('analytics_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n✅ Comprehensive analysis completed!")
        print(f"📊 Results saved to analytics_results.json")
        
        return results

def main():
    """Main execution function"""
    # Initialize analytics engine
    engine = WrapStatAnalyticsEngine('enhanced_wrapstat_training_data.csv')
    
    # Run comprehensive analysis
    results = engine.run_comprehensive_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("📋 ANALYTICS SUMMARY")
    print("="*60)
    
    print(f"📊 Data: {results['data_summary']['total_responses']} responses analyzed")
    print(f"🎯 Model Performance: R² = {results['predictive_model']['r2_score']:.3f}")
    print(f"👥 Segments: {len(results['clustering'])} participant groups identified")
    print(f"🚨 Anomalies: {results['anomalies']['negative_knowledge_change']['count']} negative changes detected")
    print(f"💡 Recommendations: {len(results['recommendations'])} improvement opportunities")
    
    # Example prediction
    print("\n" + "="*60)
    print("🔮 EXAMPLE PREDICTION")
    print("="*60)
    
    example_features = [4.0, 4.0, 4.0, 3.5, 4.5, 1.5, 2.0, 4.0, 4.0]  # Example input
    prediction = engine.predict_knowledge_improvement(example_features)
    print(f"Predicted improvement: {prediction['prediction']:.2f} points")
    print(f"Interpretation: {prediction['interpretation']}")
    
    return results

if __name__ == "__main__":
    results = main()