import streamlit as st
import requests
import json
import plotly.graph_objects as go

# Flask service URL
FLASK_API_URL = "http://127.0.0.1:5000/analyze"

# Streamlit App
def main():
    st.title("Sentiment Analysis Application")


    # User input section
    st.subheader("Enter text or upload a file for sentiment analysis")
    
    # Text input from user
    user_input = st.text_area("Enter your text here", "")

    # File uploader
    uploaded_file = st.file_uploader("OR Upload a text file", type="txt")
      
    # Dropdown for model selection
    model = st.selectbox("Select a Sentiment Analysis Model", ("VADER", "SentiWordNet", "Keras"))  
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode('utf-8')
        st.text_area("File content", user_input)

    # Analyze sentiment
    if st.button("Analyze Sentiment"):
        if user_input:
            # Prepare the request payload
            payload = {
                'text': user_input,
                'model': model
            }
            
            # Send request to Flask service
            try:
                response = requests.post(FLASK_API_URL, json=payload)
                result = response.json()

                # Extract sentiment scores from response
                sentiment_scores = result['sentiment']
                compound_score = sentiment_scores['compound'] 
                
                # Display sentiment analysis results
                #st.subheader(f"Sentiment Analysis Results (Model: {model})")
                #st.write(f"Compound: {sentiment_scores['compound']*100:.2f}%")
                #st.write(f"Positive: {sentiment_scores['pos']*100:.2f}%")
                #st.write(f"Neutral: {sentiment_scores['neu']*100:.2f}%")
                #st.write(f"Negative: {sentiment_scores['neg']*100:.2f}%")
                
                # Determine sentiment category
                if compound_score > 0:
                    sentiment_label = "Positive"
                    bar_color = "green"
                    bar_value = 0.8
                elif compound_score < 0:
                    sentiment_label = "Negative"
                    bar_color = "red"
                    bar_value = 0.8  # Use the absolute value for display
                else:
                    sentiment_label = "Neutral"
                    bar_color = "orange"
                    bar_value = 0.8  # Display a fixed value for neutral

                # Display sentiment analysis results
                st.subheader(f"Sentiment Analysis Result: (Model: {model})")

                # Create a thin horizontal bar chart without a title
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[bar_value],
                    y=[sentiment_label],
                    orientation='h',
                    marker_color=bar_color,
                    text=[sentiment_label],
                    textposition='inside',
                    insidetextanchor='middle', 
                    textfont=dict(size=18)
                ))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title=None,
                    yaxis_title=None,
                    yaxis_visible=False,
                    xaxis_visible=False,
                    height=50,  # Thin bar
                    showlegend=False
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig)
                
                # Prepare data for the Plotly bar chart
                categories = ['Compound','Positive', 'Neutral', 'Negative']
                values = [sentiment_scores['compound'], sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']]
                colors = ['blue', 'green', 'orange', 'red']  # Colors for each category

                # Create a Plotly bar chart
                fig = go.Figure([go.Bar(x=categories, y=values, marker_color=colors)])
                fig.update_layout(
                    title="Sentiment Distribution",
                    xaxis_title="Sentiment",
                    yaxis_title="Score",
                    yaxis_range=[-1, 1]  # Scale the Y-axis to -1 to 1 for compound score
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please provide some text for analysis.")

if __name__ == '__main__':
    main()
