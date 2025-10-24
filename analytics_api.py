from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from analytics_engine import WrapStatAnalyticsEngine
import threading
import time
import os

app = Flask(__name__)
CORS(app)

# Global analytics engine
analytics_engine = None
cached_results = {}
last_update = None

def initialize_analytics():
    """Initialize the analytics engine"""
    global analytics_engine, cached_results, last_update
    
    print("🔄 Initializing analytics engine...")
    analytics_engine = WrapStatAnalyticsEngine('enhanced_wrapstat_training_data.csv')
    
    # Run initial analysis
    cached_results = analytics_engine.run_comprehensive_analysis()
    last_update = time.time()
    
    print("✅ Analytics engine initialized and ready!")

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    try:
        with open('advanced_analytics_dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Dashboard file not found. Please ensure advanced_analytics_dashboard.html exists."

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'analytics_engine_loaded': analytics_engine is not None,
        'last_update': last_update,
        'cache_age_seconds': time.time() - last_update if last_update else None
    })

@app.route('/api/data/summary')
def get_data_summary():
    """Get basic data summary"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    return jsonify(cached_results.get('data_summary', {}))

@app.route('/api/model/performance')
def get_model_performance():
    """Get predictive model performance metrics"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    return jsonify(cached_results.get('predictive_model', {}))

@app.route('/api/clusters')
def get_clusters():
    """Get clustering analysis results"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    clusters = cached_results.get('clustering', [])
    
    # Format for frontend consumption
    formatted_clusters = []
    for cluster in clusters:
        formatted_clusters.append({
            'id': cluster['cluster_id'],
            'name': cluster['cluster_name'],
            'size': cluster['size'],
            'metrics': {
                'initial_knowledge': round(cluster['avg_knowledge_baseline'], 2),
                'knowledge_improvement': round(cluster['avg_knowledge_improvement'], 2),
                'satisfaction': round(cluster['avg_satisfaction'], 2),
                'duration_minutes': round(cluster['avg_duration'], 2)
            },
            'characteristics': cluster['characteristics']
        })
    
    return jsonify(formatted_clusters)

@app.route('/api/correlations')
def get_correlations():
    """Get correlation analysis"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    return jsonify(cached_results.get('correlations', {}))

@app.route('/api/recommendations')
def get_recommendations():
    """Get AI-powered recommendations"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    recommendations = cached_results.get('recommendations', [])
    
    # Format recommendations with action items
    formatted_recs = []
    for rec in recommendations:
        priority_colors = {
            'High': '#f15b28',
            'Medium': '#efbe1b',
            'Low': '#77b8e0'
        }
        
        formatted_recs.append({
            'feature': rec['feature'],
            'current_score': round(rec['current_score'], 2),
            'improvement_potential': round(rec['improvement_potential'], 3),
            'priority': rec['priority'],
            'priority_color': priority_colors.get(rec['priority'], '#1c4585'),
            'action_item': generate_action_item(rec['feature'], rec['priority'])
        })
    
    return jsonify(formatted_recs)

@app.route('/api/anomalies')
def get_anomalies():
    """Get anomaly detection results"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    return jsonify(cached_results.get('anomalies', {}))

@app.route('/api/predict', methods=['POST'])
def predict_knowledge_improvement():
    """Predict knowledge improvement for given inputs"""
    if not analytics_engine:
        return jsonify({'error': 'Analytics engine not initialized'}), 500
    
    try:
        data = request.json
        
        # Extract features in expected order
        features = [
            float(data.get('content_engaging', 4.0)),
            float(data.get('content_relevant', 4.0)),
            float(data.get('content_understandable', 4.0)),
            float(data.get('content_interactive', 3.5)),
            float(data.get('visual_support', 4.5)),
            float(data.get('training_duration', 1.5)),
            float(data.get('knowledge_baseline', 3.0)),
            float(data.get('tech_user_friendly', 4.0)),
            float(data.get('tech_easy_access', 4.0))
        ]
        
        prediction = analytics_engine.predict_knowledge_improvement(features)
        
        return jsonify({
            'prediction': round(prediction['prediction'], 2),
            'confidence': round(prediction['confidence'], 3),
            'interpretation': prediction['interpretation'],
            'inputs': {
                'content_engaging': features[0],
                'content_relevant': features[1],
                'content_understandable': features[2],
                'content_interactive': features[3],
                'visual_support': features[4],
                'training_duration': features[5],
                'knowledge_baseline': features[6],
                'tech_user_friendly': features[7],
                'tech_easy_access': features[8]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights/realtime')
def get_realtime_insights():
    """Get real-time insights and alerts"""
    if not cached_results:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    insights = []
    
    # Generate dynamic insights based on current data
    data_summary = cached_results.get('data_summary', {})
    model_perf = cached_results.get('predictive_model', {})
    anomalies = cached_results.get('anomalies', {})
    
    # Performance insight
    r2_score = model_perf.get('r2_score', 0)
    if r2_score > 0.15:
        insights.append({
            'type': 'success',
            'title': 'Model Performance',
            'message': f'Predictive model shows good performance (R² = {r2_score:.3f})',
            'priority': 'info'
        })
    elif r2_score < 0.1:
        insights.append({
            'type': 'warning',
            'title': 'Model Performance',
            'message': 'Model performance could be improved with more data or feature engineering',
            'priority': 'medium'
        })
    
    # Anomaly insight
    neg_changes = anomalies.get('negative_knowledge_change', {}).get('count', 0)
    if neg_changes > 0:
        insights.append({
            'type': 'alert',
            'title': 'Learning Issues Detected',
            'message': f'{neg_changes} participants showed negative knowledge change - requires investigation',
            'priority': 'high'
        })
    
    # Satisfaction insight
    avg_satisfaction = data_summary.get('avg_satisfaction', 0)
    if avg_satisfaction > 4.0:
        insights.append({
            'type': 'success',
            'title': 'High Satisfaction',
            'message': f'Average satisfaction is excellent ({avg_satisfaction:.2f}/5)',
            'priority': 'info'
        })
    elif avg_satisfaction < 3.5:
        insights.append({
            'type': 'warning',
            'title': 'Satisfaction Concern',
            'message': f'Average satisfaction is below target ({avg_satisfaction:.2f}/5)',
            'priority': 'medium'
        })
    
    return jsonify({
        'insights': insights,
        'timestamp': time.time(),
        'data_freshness': time.time() - last_update if last_update else None
    })

@app.route('/api/dashboard/config')
def get_dashboard_config():
    """Get dashboard configuration and metadata"""
    return jsonify({
        'title': 'WrapStat Advanced Analytics Dashboard',
        'version': '2.0.0',
        'features': {
            'predictive_analytics': True,
            'clustering': True,
            'correlation_analysis': True,
            'anomaly_detection': True,
            'realtime_predictions': True,
            'ai_recommendations': True
        },
        'data_source': 'enhanced_wrapstat_training_data.csv',
        'model_types': ['Random Forest', 'K-Means Clustering'],
        'update_frequency': 'Real-time',
        'last_model_training': last_update
    })

def generate_action_item(feature, priority):
    """Generate specific action items for recommendations"""
    action_items = {
        'content_interactive_num': {
            'High': 'Add interactive quizzes, simulations, and hands-on exercises to increase engagement',
            'Medium': 'Include more clickable elements and interactive checkpoints',
            'Low': 'Consider adding optional interactive content'
        },
        'content_engaging_num': {
            'High': 'Redesign content with storytelling, real-world examples, and multimedia elements',
            'Medium': 'Add more visual interest and varied content formats',
            'Low': 'Review content tone and presentation style'
        },
        'content_understandable_num': {
            'High': 'Simplify language, add glossaries, and improve content structure',
            'Medium': 'Add more explanations and examples for complex concepts',
            'Low': 'Review technical terminology and jargon usage'
        },
        'visual_support_num': {
            'High': 'Enhance with infographics, diagrams, and visual aids',
            'Medium': 'Add more screenshots and visual examples',
            'Low': 'Review existing visuals for clarity and relevance'
        }
    }
    
    return action_items.get(feature, {}).get(priority, 'Review and optimize this feature area')

def refresh_analytics():
    """Periodically refresh analytics data"""
    global cached_results, last_update
    
    while True:
        try:
            time.sleep(300)  # Refresh every 5 minutes
            if analytics_engine:
                print("🔄 Refreshing analytics cache...")
                cached_results = analytics_engine.run_comprehensive_analysis()
                last_update = time.time()
                print("✅ Analytics cache refreshed")
        except Exception as e:
            print(f"❌ Error refreshing analytics: {e}")

if __name__ == '__main__':
    # Initialize analytics
    initialize_analytics()
    
    # Start background refresh thread
    refresh_thread = threading.Thread(target=refresh_analytics, daemon=True)
    refresh_thread.start()
    
    print("\n" + "="*60)
    print("🚀 WrapStat Analytics API Server Starting...")
    print("="*60)
    print("📊 Dashboard: http://localhost:5000")
    print("🔗 API Health: http://localhost:5000/api/health")
    print("💡 Recommendations: http://localhost:5000/api/recommendations")
    print("👥 Clusters: http://localhost:5000/api/clusters")
    print("="*60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)