"""
AMEEMAW - AI-Mediated Educational Experience for Mammary Awareness
A breast ultrasound education app with Nana, your caring AI companion.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from utils.model import load_model, predict_class, get_class_info
from utils.gradcam import generate_gradcam, overlay_gradcam
from utils.nana import NanaCompanion, get_encouraging_message

# Page Configuration
st.set_page_config(
    page_title="AMEEMAW - Learn with Nana",
    page_icon="💝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for warm, comforting UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&family=Playfair+Display:wght@500;600&display=swap');
    
    /* Main theme colors - warm, comforting palette */
    :root {
        --primary-rose: #E8B4B8;
        --soft-cream: #FDF6F0;
        --warm-mauve: #C9A9A6;
        --gentle-sage: #B5C4B1;
        --deep-plum: #5D4E60;
        --sunshine: #F7E1AE;
        --trust-blue: #7BA3B7;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, var(--soft-cream) 0%, #FEF9F3 50%, #F8EDE3 100%);
        font-family: 'Nunito', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: var(--deep-plum) !important;
    }
    
    /* Nana's message bubbles */
    .nana-bubble {
        background: linear-gradient(135deg, #FFFFFF 0%, var(--soft-cream) 100%);
        border: 2px solid var(--primary-rose);
        border-radius: 20px 20px 20px 5px;
        padding: 20px 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(232, 180, 184, 0.3);
        position: relative;
    }
    
    .nana-bubble::before {
        content: "🌸";
        position: absolute;
        top: -10px;
        left: 15px;
        font-size: 20px;
    }
    
    /* Cards for modes */
    .mode-card {
        background: white;
        border-radius: 25px;
        padding: 30px;
        box-shadow: 0 8px 30px rgba(93, 78, 96, 0.1);
        border: 1px solid var(--primary-rose);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .mode-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(93, 78, 96, 0.15);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-rose) 0%, var(--warm-mauve) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(232, 180, 184, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(232, 180, 184, 0.5);
    }
    
    /* Result boxes */
    .result-normal {
        background: linear-gradient(135deg, var(--gentle-sage) 0%, #C8D5C4 100%);
        border-radius: 20px;
        padding: 25px;
        border-left: 5px solid #7DA67A;
    }
    
    .result-benign {
        background: linear-gradient(135deg, var(--sunshine) 0%, #FAE9C4 100%);
        border-radius: 20px;
        padding: 25px;
        border-left: 5px solid #D4A84B;
    }
    
    .result-malignant {
        background: linear-gradient(135deg, var(--primary-rose) 0%, #DFADB0 100%);
        border-radius: 20px;
        padding: 25px;
        border-left: 5px solid #C48B8F;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 3px dashed var(--primary-rose);
        border-radius: 20px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.7);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--soft-cream) 0%, white 100%);
    }
    
    /* Hope message */
    .hope-message {
        background: linear-gradient(135deg, #F7E1AE 0%, #FCF0D8 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        font-style: italic;
        color: var(--deep-plum);
        border: 1px solid #E8D4A8;
    }
    
    /* Progress indicator */
    .learning-progress {
        background: white;
        border-radius: 30px;
        padding: 8px 20px;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Image container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 3px solid white;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: var(--warm-mauve);
        font-size: 14px;
        margin-top: 40px;
        padding: 20px;
    }
    
    /* Confidence meter */
    .confidence-meter {
        height: 12px;
        border-radius: 6px;
        background: #E8E8E8;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    /* Accessibility improvements */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        border: 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'nana' not in st.session_state:
        st.session_state.nana = NanaCompanion()
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    if 'user_guess' not in st.session_state:
        st.session_state.user_guess = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'show_result' not in st.session_state:
        st.session_state.show_result = False
    if 'learning_streak' not in st.session_state:
        st.session_state.learning_streak = 0
    if 'model' not in st.session_state:
        st.session_state.model = None


def display_header():
    """Display the app header with Nana's welcome."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0;">💝 AMEEMAW</h1>
            <p style="font-size: 1.2rem; color: #C9A9A6; margin-top: 10px;">
                AI-Mediated Educational Experience for Mammary Awareness
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_nana_message(message: str, message_type: str = "greeting"):
    """Display a message from Nana in a styled bubble."""
    icon = "👵🏽" if message_type == "greeting" else "💭"
    st.markdown(f"""
    <div class="nana-bubble">
        <strong style="color: #5D4E60; font-size: 1.1rem;">{icon} Nana says:</strong>
        <p style="margin-top: 10px; color: #5D4E60; line-height: 1.6;">{message}</p>
    </div>
    """, unsafe_allow_html=True)


def display_hope_message():
    """Display a subtle faith-based message of hope."""
    messages = [
        "✨ 'For I know the plans I have for you... plans to give you hope and a future.' — May knowledge bring you peace.",
        "🌸 Every step in learning is a step toward hope. You're doing wonderfully.",
        "💫 'Be strong and courageous.' Knowledge is power, and you're gaining it every day.",
        "🌿 In moments of uncertainty, remember: understanding brings comfort. You are not alone.",
        "🕊️ 'Peace I leave with you.' May your learning journey bring clarity and calm.",
    ]
    message = np.random.choice(messages)
    st.markdown(f"""
    <div class="hope-message">
        {message}
    </div>
    """, unsafe_allow_html=True)


def mode_selection():
    """Display mode selection cards."""
    st.markdown("### Choose Your Learning Path 📚")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mode-card">
            <h3 style="color: #5D4E60;">🎓 Learn Mode</h3>
            <p style="color: #888;">
                Test your knowledge! Upload an ultrasound image, 
                make your guess, then see how you did. 
                Nana will guide you every step of the way.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Learning 🌟", key="learn_btn"):
            st.session_state.current_mode = "learn"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="mode-card">
            <h3 style="color: #5D4E60;">🔍 Explain Mode</h3>
            <p style="color: #888;">
                Get detailed explanations with BI-RADS classifications 
                and visual heatmaps showing what the AI sees.
                Perfect for deeper understanding.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Explanations 💡", key="explain_btn"):
            st.session_state.current_mode = "explain"
            st.rerun()


def learn_mode():
    """Interactive learning mode with guess-then-reveal."""
    st.markdown("## 🎓 Learn Mode")
    
    # Welcome message
    welcome = st.session_state.nana.get_response(
        "learning_welcome",
        context={"streak": st.session_state.learning_streak}
    )
    display_nana_message(welcome, "greeting")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a breast ultrasound image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a grayscale breast ultrasound image for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Your Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### Make Your Guess")
            
            if not st.session_state.show_result:
                guess_prompt = st.session_state.nana.get_response(
                    "guess_prompt",
                    context={}
                )
                display_nana_message(guess_prompt, "thinking")
                
                st.markdown("**What do you think this shows?**")
                
                guess_col1, guess_col2, guess_col3 = st.columns(3)
                
                with guess_col1:
                    if st.button("🟢 Normal", key="guess_normal", use_container_width=True):
                        st.session_state.user_guess = "Normal"
                        st.session_state.show_result = True
                        st.rerun()
                
                with guess_col2:
                    if st.button("🟡 Benign", key="guess_benign", use_container_width=True):
                        st.session_state.user_guess = "Benign"
                        st.session_state.show_result = True
                        st.rerun()
                
                with guess_col3:
                    if st.button("🔴 Malignant", key="guess_malignant", use_container_width=True):
                        st.session_state.user_guess = "Malignant"
                        st.session_state.show_result = True
                        st.rerun()
            
            else:
                # Show results
                with st.spinner("Nana is analyzing... 🔍"):
                    # Load model if needed
                    if st.session_state.model is None:
                        st.session_state.model = load_model()
                    
                    # Get prediction
                    predicted_class, confidence, probabilities = predict_class(
                        st.session_state.model, image
                    )
                    
                    # Generate Grad-CAM
                    gradcam_heatmap = generate_gradcam(st.session_state.model, image)
                    overlay_image = overlay_gradcam(image, gradcam_heatmap)
                
                # Display result
                is_correct = st.session_state.user_guess == predicted_class
                
                if is_correct:
                    st.session_state.learning_streak += 1
                    result_class = "result-normal" if predicted_class == "Normal" else \
                                   "result-benign" if predicted_class == "Benign" else "result-malignant"
                    
                    st.markdown(f"""
                    <div class="{result_class}">
                        <h3>🎉 Wonderful! You got it right!</h3>
                        <p>The AI agrees: <strong>{predicted_class}</strong> (Confidence: {confidence:.1%})</p>
                        <p>🔥 Learning streak: {st.session_state.learning_streak}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    feedback = st.session_state.nana.get_response(
                        "correct_guess",
                        context={"predicted": predicted_class, "streak": st.session_state.learning_streak}
                    )
                else:
                    st.session_state.learning_streak = 0
                    st.markdown(f"""
                    <div class="result-benign">
                        <h3>💭 Learning Opportunity!</h3>
                        <p>You guessed <strong>{st.session_state.user_guess}</strong>, 
                        but the AI predicts <strong>{predicted_class}</strong> (Confidence: {confidence:.1%})</p>
                        <p>That's okay, sweetie! Every guess teaches us something new. 💕</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    feedback = st.session_state.nana.get_response(
                        "incorrect_guess",
                        context={
                            "user_guess": st.session_state.user_guess,
                            "predicted": predicted_class
                        }
                    )
                
                display_nana_message(feedback, "feedback")
                
                # Show Grad-CAM
                st.markdown("### What the AI Focused On")
                st.image(overlay_image, caption="Grad-CAM Visualization", use_container_width=True)
                
                # Get explanation
                class_info = get_class_info(predicted_class)
                explanation = st.session_state.nana.get_response(
                    "explain_finding",
                    context={
                        "predicted": predicted_class,
                        "confidence": confidence,
                        "class_info": class_info
                    }
                )
                display_nana_message(explanation, "teaching")
                
                # Display confidence breakdown
                st.markdown("### Confidence Breakdown")
                for cls, prob in probabilities.items():
                    color = "#B5C4B1" if cls == "Normal" else "#F7E1AE" if cls == "Benign" else "#E8B4B8"
                    st.markdown(f"""
                    <div style="margin: 8px 0;">
                        <span>{cls}: {prob:.1%}</span>
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: {prob*100}%; background: {color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Reset button
                if st.button("Try Another Image 🔄", key="reset_learn"):
                    st.session_state.user_guess = None
                    st.session_state.show_result = False
                    st.rerun()
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Home", key="back_learn"):
        st.session_state.current_mode = None
        st.session_state.user_guess = None
        st.session_state.show_result = False
        st.rerun()


def explain_mode():
    """Detailed explanation mode with BI-RADS."""
    st.markdown("## 🔍 Explain Mode")
    
    # Welcome message
    welcome = st.session_state.nana.get_response("explain_welcome", context={})
    display_nana_message(welcome, "greeting")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a breast ultrasound image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a grayscale breast ultrasound image for detailed analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        with st.spinner("Nana is carefully analyzing this image... 🔍"):
            # Load model if needed
            if st.session_state.model is None:
                st.session_state.model = load_model()
            
            # Get prediction
            predicted_class, confidence, probabilities = predict_class(
                st.session_state.model, image
            )
            
            # Generate Grad-CAM
            gradcam_heatmap = generate_gradcam(st.session_state.model, image)
            overlay_image = overlay_gradcam(image, gradcam_heatmap)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### AI Focus Areas (Grad-CAM)")
            st.image(overlay_image, use_container_width=True)
        
        # Result display
        result_class = "result-normal" if predicted_class == "Normal" else \
                       "result-benign" if predicted_class == "Benign" else "result-malignant"
        
        st.markdown(f"""
        <div class="{result_class}">
            <h3>Classification: {predicted_class}</h3>
            <p>Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # BI-RADS Explanation
        st.markdown("### BI-RADS Classification Guide")
        
        class_info = get_class_info(predicted_class)
        birads_explanation = st.session_state.nana.get_response(
            "birads_explanation",
            context={
                "predicted": predicted_class,
                "confidence": confidence,
                "class_info": class_info
            }
        )
        display_nana_message(birads_explanation, "teaching")
        
        # Detailed BI-RADS table
        st.markdown("""
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <tr style="background: #E8B4B8; color: white;">
                <th style="padding: 12px; text-align: left;">BI-RADS Category</th>
                <th style="padding: 12px; text-align: left;">Assessment</th>
                <th style="padding: 12px; text-align: left;">Recommendation</th>
            </tr>
            <tr style="background: #F8F8F8;">
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">0</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Incomplete</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Additional imaging needed</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">1</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Negative</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Routine screening</td>
            </tr>
            <tr style="background: #F8F8F8;">
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">2</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Benign</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Routine screening</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">3</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Probably Benign</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Short-term follow-up</td>
            </tr>
            <tr style="background: #F8F8F8;">
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">4</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Suspicious</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Biopsy recommended</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">5</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Highly Suspicious</td>
                <td style="padding: 12px; border-bottom: 1px solid #ddd;">Biopsy required</td>
            </tr>
            <tr style="background: #F8F8F8;">
                <td style="padding: 12px;">6</td>
                <td style="padding: 12px;">Known Malignancy</td>
                <td style="padding: 12px;">Treatment planning</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("### Detailed Probability Analysis")
        for cls, prob in probabilities.items():
            color = "#B5C4B1" if cls == "Normal" else "#F7E1AE" if cls == "Benign" else "#E8B4B8"
            st.markdown(f"""
            <div style="margin: 10px 0; padding: 15px; background: white; border-radius: 10px; border-left: 4px solid {color};">
                <strong>{cls}</strong>: {prob:.1%}
                <div class="confidence-meter" style="margin-top: 8px;">
                    <div class="confidence-fill" style="width: {prob*100}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Educational note
        educational_note = st.session_state.nana.get_response(
            "educational_note",
            context={"predicted": predicted_class}
        )
        display_nana_message(educational_note, "teaching")
        
        # Hope message
        display_hope_message()
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Home", key="back_explain"):
        st.session_state.current_mode = None
        st.rerun()


def display_sidebar():
    """Display the sidebar with information and settings."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #5D4E60;">👵🏽 Meet Nana</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border-radius: 15px; padding: 20px; margin: 10px 0;">
            <p style="color: #666; line-height: 1.6;">
                Hello, dear! I'm Nana, your caring guide through breast ultrasound education. 
                I'm here to help you learn in a warm, supportive environment. 
                Remember, knowledge is power! 💕
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Your Progress")
        st.markdown(f"""
        <div class="learning-progress">
            🔥 Current Streak: {st.session_state.learning_streak}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About AMEEMAW")
        st.markdown("""
        <div style="font-size: 14px; color: #666; line-height: 1.6;">
            <p><strong>AMEEMAW</strong> is an educational tool designed to help 
            healthcare students and professionals learn breast ultrasound interpretation.</p>
            
            <p>🔬 <strong>Technology:</strong><br>
            ResNet-50 deep learning model with Grad-CAM visualization</p>
            
            <p>📚 <strong>Classifications:</strong><br>
            Normal, Benign, Malignant</p>
            
            <p>⚠️ <strong>Disclaimer:</strong><br>
            This is an educational tool only. It should not be used for clinical diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Hope message in sidebar
        st.markdown("""
        <div style="background: linear-gradient(135deg, #F7E1AE 0%, #FCF0D8 100%); 
                    border-radius: 15px; padding: 15px; text-align: center; font-style: italic;">
            🕊️ "May your learning journey bring you peace and understanding."
        </div>
        """, unsafe_allow_html=True)


def display_footer():
    """Display the footer."""
    st.markdown("""
    <div class="footer-text">
        <p>Made with 💝 for education and awareness</p>
        <p style="font-size: 12px; color: #AAA;">
            AMEEMAW v1.0 | Educational Use Only | Not for Clinical Diagnosis
        </p>
        <p style="font-size: 11px; color: #BBB; margin-top: 10px;">
            🌸 "For where two or three gather together, there am I with them." — May knowledge and compassion guide your learning.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    init_session_state()
    display_sidebar()
    display_header()
    
    if st.session_state.current_mode is None:
        # Welcome message from Nana
        welcome_message = st.session_state.nana.get_response("welcome", context={})
        display_nana_message(welcome_message, "greeting")
        
        mode_selection()
        display_hope_message()
    
    elif st.session_state.current_mode == "learn":
        learn_mode()
    
    elif st.session_state.current_mode == "explain":
        explain_mode()
    
    display_footer()


if __name__ == "__main__":
    main()
