"""
WrapStat Advanced Analytics Dashboard Launcher
==============================================

This script provides multiple ways to run the advanced analytics dashboard:

1. Start Real-time API Server (Recommended)
2. Open Static Dashboard
3. Run Analytics Engine Only
4. Generate Demo Data

Usage:
    python launcher.py [option]
    
Options:
    --api      Start the Flask API server with real-time dashboard
    --static   Open the static HTML dashboard  
    --engine   Run analytics engine only
    --demo     Generate demo data for testing
    --help     Show this help message
"""

import sys
import subprocess
import webbrowser
import os
import time

def main():
    if len(sys.argv) == 1 or '--help' in sys.argv:
        print(__doc__)
        return
    
    option = sys.argv[1]
    
    if option == '--api':
        start_api_server()
    elif option == '--static':
        open_static_dashboard()
    elif option == '--engine':
        run_analytics_engine()
    elif option == '--demo':
        generate_demo_data()
    else:
        print(f"Unknown option: {option}")
        print("Use --help for available options")

def start_api_server():
    """Start the Flask API server with real-time dashboard"""
    print("🚀 Starting WrapStat Advanced Analytics API Server...")
    print("="*60)
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'analytics_api.py'], check=True)
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except FileNotFoundError:
        print("❌ analytics_api.py not found. Please ensure the file exists.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def open_static_dashboard():
    """Open the static HTML dashboard"""
    print("📊 Opening Static Dashboard...")
    
    dashboard_files = [
        'realtime_dashboard.html',
        'advanced_analytics_dashboard.html', 
        'dashboard_example.html'
    ]
    
    for dashboard_file in dashboard_files:
        if os.path.exists(dashboard_file):
            file_path = os.path.abspath(dashboard_file)
            webbrowser.open(f'file://{file_path}')
            print(f"✅ Opened {dashboard_file} in browser")
            return
    
    print("❌ No dashboard files found. Please ensure HTML files exist.")

def run_analytics_engine():
    """Run the analytics engine only"""
    print("🧠 Running Analytics Engine...")
    
    try:
        subprocess.run([sys.executable, 'analytics_engine.py'], check=True)
        print("✅ Analytics completed successfully")
    except FileNotFoundError:
        print("❌ analytics_engine.py not found")
    except Exception as e:
        print(f"❌ Error running analytics: {e}")

def generate_demo_data():
    """Generate demo data for testing"""
    print("🎲 Generating Demo Data...")
    
    # This would create sample data for demonstration
    demo_script = """
import pandas as pd
import numpy as np

# Generate demo survey data
np.random.seed(42)
n_responses = 50

demo_data = {
    'response_id': [f'R_demo_{i:03d}' for i in range(n_responses)],
    'knowledge_before_numeric': np.random.choice([1, 2, 3, 4, 5], n_responses, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'knowledge_after_numeric': np.random.choice([2, 3, 4, 5], n_responses, p=[0.1, 0.3, 0.4, 0.2]),
    'content_engaging_numeric': np.random.choice([3, 4, 5], n_responses, p=[0.2, 0.5, 0.3]),
    'content_relevant_numeric': np.random.choice([3, 4, 5], n_responses, p=[0.1, 0.6, 0.3]),
    'content_understandable_numeric': np.random.choice([3, 4, 5], n_responses, p=[0.15, 0.5, 0.35]),
    'content_interactive_numeric': np.random.choice([2, 3, 4, 5], n_responses, p=[0.1, 0.3, 0.4, 0.2]),
    'content_visual_support_numeric': np.random.choice([3, 4, 5], n_responses, p=[0.1, 0.4, 0.5]),
    'training_duration_minutes': np.random.normal(1.5, 0.5, n_responses),
    'region': np.random.choice(['Northern Illinois', 'Central Illinois', 'Southern Illinois'], n_responses)
}

df = pd.DataFrame(demo_data)
df['knowledge_improvement'] = df['knowledge_after_numeric'] - df['knowledge_before_numeric']
df['training_duration_minutes'] = np.clip(df['training_duration_minutes'], 0.5, 5.0)

df.to_csv('demo_wrapstat_data.csv', index=False)
print(f"✅ Generated demo data: {len(df)} responses")
print(f"📁 Saved to: demo_wrapstat_data.csv")
"""
    
    with open('generate_demo.py', 'w') as f:
        f.write(demo_script)
    
    try:
        subprocess.run([sys.executable, 'generate_demo.py'], check=True)
        os.remove('generate_demo.py')  # Clean up
        print("✅ Demo data generated successfully")
    except Exception as e:
        print(f"❌ Error generating demo data: {e}")

if __name__ == '__main__':
    main()