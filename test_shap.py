import pandas as pd
import plotly.graph_objects as go

def test_chart():
    df = pd.DataFrame({
        'Feature': ['Age', 'HIV', 'Education'],
        'SHAP Value': [0.5, 0.3, -0.2]
    }).sort_values('SHAP Value', ascending=True)

    colors = ['#E74C3C' if v > 0 else '#2E75B6' for v in df['SHAP Value']]

    fig = go.Figure(go.Bar(
        x=df['SHAP Value'],
        y=df['Feature'],
        orientation='h',
        marker_color=colors,
    ))

    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    print("HTML length:", len(html))
    print("First 200 chars:", html[:200])

test_chart()
