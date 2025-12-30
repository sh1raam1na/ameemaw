"""
AMEEMAW - AI Mamma-Echography Educator & Empathetic Mentor for Awareness & Wellness
A breast ultrasound education app with Nana, your caring AI companion.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from utils.model import load_model, predict_class, get_class_info

# Model metrics for bias display (from 05_bias_audit.ipynb)
MODEL_METRICS = {
    'overall_accuracy': 0.846,
    'malignant_recall': 0.839,
    'malignant_precision': 0.722,
    'benign_recall': 0.818,
    'normal_recall': 0.950,
    'small_malignant_recall': 0.50,
    'large_lesion_accuracy': 0.708,
    'med_light_brightness_accuracy': 0.759,
}
from utils.gradcam import generate_gradcam, overlay_gradcam
from utils.nana import NanaCompanion, get_encouraging_message

# Page Configuration
st.set_page_config(
    page_title="AMEEMAW - Learn with Nana",
    page_icon="🎀",
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
    
    /* Chat bubble style for Explain Mode */
    .chat-bubble-nana {
        background: linear-gradient(135deg, #FFFFFF 0%, var(--soft-cream) 100%);
        border: 2px solid var(--primary-rose);
        border-radius: 20px 20px 20px 5px;
        padding: 20px 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(232, 180, 184, 0.3);
        display: flex;
        align-items: flex-start;
        gap: 15px;
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, var(--trust-blue) 0%, #8FB3C4 100%);
        border-radius: 20px 20px 5px 20px;
        padding: 15px 20px;
        margin: 15px 0;
        margin-left: auto;
        max-width: 80%;
        color: white;
    }
    
    .nana-avatar {
        font-size: 3rem;
        line-height: 1;
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
    
    /* Bias info box */
    .bias-info-box {
        background: #FFF9E6;
        border: 1px solid #E8D4A8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-size: 14px;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border: 2px solid var(--primary-rose) !important;
        border-radius: 15px !important;
        padding: 12px 15px !important;
        background: white !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--warm-mauve) !important;
        box-shadow: 0 0 5px rgba(232, 180, 184, 0.5) !important;
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
    if 'correct_guesses' not in st.session_state:
        st.session_state.correct_guesses = 0
    if 'total_attempts' not in st.session_state:
        st.session_state.total_attempts = 0
    if 'result_counted' not in st.session_state:
        st.session_state.result_counted = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'uploaded_file_id' not in st.session_state:
        st.session_state.uploaded_file_id = None
    if 'explain_mode_state' not in st.session_state:
        st.session_state.explain_mode_state = 'welcome'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'birads_score' not in st.session_state:
        st.session_state.birads_score = None


def display_header():
    """Display the app header with Nana's welcome."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0;">🎀 AMEEMAW</h1>
            <p style="font-size: 1.2rem; color: #C9A9A6; margin-top: 10px;">
                AI Mamma-Echography Educator & Empathetic Mentor for Awareness & Wellness
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_nana_message(message: str, message_type: str = "greeting"):
    """Display a message from Nana in a styled bubble."""
    import re
    # Convert **text** to <strong>text</strong> for HTML rendering
    message_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', message)
    # Convert newlines to <br> for proper line breaks
    message_html = message_html.replace('\n', '<br>')
    
    icon = "👵🏽" if message_type == "greeting" else "💭"
    st.markdown(f"""
    <div class="nana-bubble">
        <strong style="color: #5D4E60; font-size: 1.1rem;">{icon} Nana says:</strong>
        <p style="margin-top: 10px; color: #5D4E60; line-height: 1.6;">{message_html}</p>
    </div>
    """, unsafe_allow_html=True)


def display_hope_message():
    """Display a subtle message of hope."""
    messages = [
        "🌸 Every step in learning is a step toward hope. You're doing wonderfully.",
        "🌿 In moments of uncertainty, remember: understanding brings comfort. You are not alone.",
        "💙 Learning together, one step at a time.",
        "🎀 Here for your questions, here for you.",
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
                Nana will explain why the AI made its prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Learning 🌟", key="learn_btn"):
            st.session_state.current_mode = "learn"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="mode-card">
            <h3 style="color: #5D4E60;">💬 Talk with Nana</h3>
            <p style="color: #666; font-size: 0.9rem;">(Explain Mode)</p>
            <p style="color: #888;">
                Have questions about BI-RADS results? Need someone to talk to 
                about breast health? Nana is here to listen and support you.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Talk with Nana 💕", key="explain_btn"):
            st.session_state.current_mode = "explain"
            st.session_state.explain_mode_state = 'welcome'
            st.rerun()


def get_educational_explanation(predicted_class: str, confidence: float) -> str:
    """Get educational explanation for why AI made this prediction."""
    explanations = {
        'Normal': """**Why the AI chose Normal:**

The AI detected features commonly associated with normal breast tissue:

- Homogeneous (uniform) echotexture throughout
- No focal masses or suspicious areas detected
- Well-organized tissue layers
- Absence of irregular shapes or margins

**What to look for:** Normal tissue appears uniform without any distinct lumps or shadows.
""",
        'Benign': """**Why the AI chose Benign:**

The AI detected features commonly associated with benign masses:

- Oval or round shape with smooth, well-defined margins
- Parallel orientation to the skin surface
- Uniform internal echoes (consistent texture inside)
- Possible thin, echogenic capsule (bright outline)
- No posterior acoustic shadowing

**What to look for:** Benign masses typically look "friendly" — smooth, round, and well-contained.
""",
        'Malignant': """**Why the AI chose Malignant:**

The AI detected features that may warrant further evaluation:

- Irregular or spiculated (spiky) shape
- Non-parallel (vertical) orientation to skin
- Heterogeneous internal echoes (varied texture)
- Posterior acoustic shadowing (dark area behind mass)
- Indistinct or angular margins
- Possible microcalcifications

**What to look for:** Concerning features often appear "unfriendly" — irregular, spiky, with shadowing.
"""
    }
    return explanations.get(predicted_class, "")


def get_bias_note(confidence: float, predicted_class: str) -> str:
    """Get bias/limitation note based on prediction."""
    notes = []
    
    if confidence < 0.50:
        notes.append("⚠️ **Very low confidence** — this image may have features our model struggles with (unusual brightness, small lesions, or edge cases).")
    elif confidence < 0.70:
        notes.append("⚠️ **Lower confidence** — model accuracy varies with image characteristics.")
    
    if predicted_class == 'Benign':
        notes.append(f"🚨 **Important:** Our model misses ~{(1-MODEL_METRICS['malignant_recall'])*100:.0f}% of malignant cases. A 'Benign' prediction should never replace professional evaluation.")
    
    return "\n\n".join(notes) if notes else ""


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
    st.markdown("**📁 Upload an image:**")
    uploaded_file = st.file_uploader(
        "Upload a breast ultrasound image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a grayscale breast ultrasound image for analysis",
        key="learn_uploader",
        label_visibility="collapsed"
    )
    st.caption("💡 *Tip: If using a screenshot, save it first (Ctrl+S) then drag & drop or browse to upload.*")
    
    # Reset state if file changed or removed
    current_file_id = uploaded_file.file_id if uploaded_file else None
    if current_file_id != st.session_state.uploaded_file_id:
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.user_guess = None
        st.session_state.show_result = False
        st.session_state.result_counted = False
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Your Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### Make Your Guess")
            
            if not st.session_state.show_result:
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
                
                st.markdown("---")
                st.markdown("*Take your time to observe the image. Look for shape, margins, and internal texture.*")
            
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
                    overlay_result = overlay_gradcam(image, gradcam_heatmap)
                    
                    # Handle both tuple and single image return
                    if isinstance(overlay_result, tuple):
                        overlay_image = overlay_result[0]
                    else:
                        overlay_image = overlay_result
                
                # Display result
                is_correct = st.session_state.user_guess == predicted_class
                
                # Track attempts (only once per result view)
                if not st.session_state.result_counted:
                    st.session_state.total_attempts += 1
                    if is_correct:
                        st.session_state.correct_guesses += 1
                        st.session_state.learning_streak += 1
                    else:
                        st.session_state.learning_streak = 0
                    st.session_state.result_counted = True
                
                if is_correct:
                    result_class = "result-normal" if predicted_class == "Normal" else \
                                   "result-benign" if predicted_class == "Benign" else "result-malignant"
                    
                    st.markdown(f"""
                    <div class="{result_class}">
                        <h3>🎉 Wonderful! You got it right!</h3>
                        <p>The AI agrees: <strong>{predicted_class}</strong> (Confidence: {confidence:.1%})</p>
                        <p>🎯 Score: {st.session_state.correct_guesses}/{st.session_state.total_attempts} | 🔥 Streak: {st.session_state.learning_streak}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    feedback = f"🎉 **Great job, dear!** You correctly identified this as **{predicted_class}**! Your learning is really paying off. Keep up the wonderful work! 💙🌸"
                else:
                    st.markdown(f"""
                    <div class="result-benign">
                        <h3>💭 Learning Opportunity!</h3>
                        <p>You guessed <strong>{st.session_state.user_guess}</strong>, 
                        but the AI predicts <strong>{predicted_class}</strong> (Confidence: {confidence:.1%})</p>
                        <p>🎯 Score: {st.session_state.correct_guesses}/{st.session_state.total_attempts}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    feedback = f"That's okay! The AI sees features suggesting **{predicted_class}**. Let's look at the Grad-CAM and explanation below to understand why. Every attempt helps you learn! 💙"
                
                display_nana_message(feedback, "feedback")
        
        # Only show detailed results if we have results to show
        if st.session_state.show_result:
            st.markdown("---")
            
            # Probability Breakdown
            st.markdown("### 📊 Probability Breakdown")
            for cls, prob in probabilities.items():
                color = "#B5C4B1" if cls == "Normal" else "#F7E1AE" if cls == "Benign" else "#E8B4B8"
                indicator = " ← AI's choice" if cls == predicted_class else ""
                st.markdown(f"""
                <div style="margin: 10px 0; padding: 15px; background: white; border-radius: 10px; border-left: 4px solid {color};">
                    <strong>{cls}</strong>: {prob:.1%}{indicator}
                    <div class="confidence-meter" style="margin-top: 8px;">
                        <div class="confidence-fill" style="width: {prob*100}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Educational Explanation
            st.markdown("### 🔍 Why This Classification?")
            explanation = get_educational_explanation(predicted_class, confidence)
            st.markdown(explanation)
            
            # Grad-CAM Visualization
            st.markdown("### 🎯 What the AI Focused On (Grad-CAM)")
            col_orig, col_gradcam = st.columns(2)
            with col_orig:
                st.image(image, caption="Original Image", use_container_width=True)
            with col_gradcam:
                st.image(overlay_image, caption="AI Attention Heatmap", use_container_width=True)
            
            st.markdown("*The heatmap shows which areas the AI focused on most (red = high attention, blue = low attention).*")
            
            # Bias Note (inline)
            bias_note = get_bias_note(confidence, predicted_class)
            if bias_note:
                st.markdown(f"""
                <div class="bias-info-box">
                    {bias_note}
                </div>
                """, unsafe_allow_html=True)
            
            # Expandable Model Limitations
            with st.expander("ℹ️ About this prediction — Model Limitations"):
                st.markdown(f"""
### 📊 Model Performance
- **Overall accuracy:** {MODEL_METRICS['overall_accuracy']:.1%}
- **Malignant recall:** {MODEL_METRICS['malignant_recall']:.1%} (misses ~{(1-MODEL_METRICS['malignant_recall'])*100:.0f}% of cancers)
- **Small lesion recall:** {MODEL_METRICS['small_malignant_recall']:.1%} (misses half of small cancers)

### ⚠️ Known Limitations
- Lower accuracy on large lesions ({MODEL_METRICS['large_lesion_accuracy']:.1%})
- Lower accuracy on medium-light brightness images ({MODEL_METRICS['med_light_brightness_accuracy']:.1%})
- Best performance on dark images (86.7%)

### 🎓 What This Means for Learning
This model is for **educational purposes only**. Real diagnosis requires professional 
radiologists who consider clinical history, multiple imaging views, and physical examination.
""")
            
            # Action buttons at the bottom
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Try Another Image", key="reset_learn", use_container_width=True):
                    st.session_state.user_guess = None
                    st.session_state.show_result = False
                    st.session_state.uploaded_file_id = None
                    st.session_state.result_counted = False
                    st.rerun()
            with col_btn2:
                if st.button("← Back to Home", key="back_learn_results", use_container_width=True):
                    st.session_state.current_mode = None
                    st.session_state.user_guess = None
                    st.session_state.show_result = False
                    st.session_state.uploaded_file_id = None
                    st.session_state.result_counted = False
                    st.rerun()
    
    # Back button (only shows when no image uploaded)
    if uploaded_file is None:
        st.markdown("---")
        if st.button("← Back to Home", key="back_learn"):
            st.session_state.current_mode = None
            st.session_state.user_guess = None
            st.session_state.show_result = False
            st.session_state.uploaded_file_id = None
            st.rerun()


def explain_mode():
    """Talk with Nana - Emotional support and BI-RADS explanation."""
    
    # Welcome screen
    if st.session_state.explain_mode_state == 'welcome':
        st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <div style="font-size: 6rem;">👵🏽</div>
            <h2 style="color: #5D4E60; margin-top: 20px;">Talk with Nana</h2>
            <p style="color: #888; font-size: 1.1rem;">(Explain Mode)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Welcome message
        st.markdown("""
        <div class="chat-bubble-nana">
            <div class="nana-avatar">👵🏽</div>
            <div>
                <p style="color: #5D4E60; line-height: 1.6; margin: 0;">
                    Welcome, sweetheart. I'm here. Would you like to talk about breast health, 
                    or just need someone to listen? 💕
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="mode-card" style="text-align: center;">
                <h3 style="color: #5D4E60;">🔢 I have a BI-RADS score</h3>
                <p style="color: #888;">
                    Let me help you understand what your doctor's BI-RADS result means.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Understand my BI-RADS", key="birads_btn", use_container_width=True):
                st.session_state.explain_mode_state = 'birads_input'
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="mode-card" style="text-align: center;">
                <h3 style="color: #5D4E60;">💬 I just want to talk</h3>
                <p style="color: #888;">
                    I'm here to listen and support you. No question is too small.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Just chat with Nana", key="chat_btn", use_container_width=True):
                st.session_state.explain_mode_state = 'chat'
                st.session_state.chat_history = []
                st.rerun()
        
        # Back to Home button
        st.markdown("---")
        if st.button("← Back to Home", key="back_explain_welcome"):
            st.session_state.current_mode = None
            st.session_state.explain_mode_state = 'welcome'
            st.rerun()
    
    # BI-RADS Input screen
    elif st.session_state.explain_mode_state == 'birads_input':
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 4rem;">👵🏽</div>
            <h2 style="color: #5D4E60;">Understanding Your BI-RADS Score</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="chat-bubble-nana">
            <div class="nana-avatar">👵🏽</div>
            <div>
                <p style="color: #5D4E60; line-height: 1.6; margin: 0;">
                    I'd be happy to help you understand your BI-RADS score, dear. 
                    What number did your doctor give you? (0-6) 💙
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        birads = st.selectbox(
            "Select your BI-RADS category:",
            options=[None, 0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: "Choose a category..." if x is None else f"BI-RADS {x}",
            key="birads_select"
        )
        
        if birads is not None:
            if st.button("Explain this to me 💕", key="explain_birads"):
                st.session_state.birads_score = birads
                st.session_state.explain_mode_state = 'birads_result'
                st.rerun()
        
        st.markdown("---")
        if st.button("← Back", key="back_birads"):
            st.session_state.explain_mode_state = 'welcome'
            st.rerun()
    
    # BI-RADS Result screen
    elif st.session_state.explain_mode_state == 'birads_result':
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 4rem;">👵🏽</div>
        </div>
        """, unsafe_allow_html=True)
        
        birads = st.session_state.birads_score
        
        # Get explanation from Nana
        explanation = st.session_state.nana.get_response(
            "birads_explanation",
            context={"birads_score": birads, "mode": "explain"}
        )
        
        st.markdown(f"""
        <div class="chat-bubble-nana">
            <div class="nana-avatar">👵🏽</div>
            <div>
                <p style="color: #5D4E60; line-height: 1.6; margin: 0;">
                    {explanation}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Follow-up options
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Ask another question", key="another_q"):
                st.session_state.explain_mode_state = 'chat'
                st.session_state.chat_history = [
                    {"role": "nana", "content": explanation}
                ]
                st.rerun()
        with col2:
            if st.button("Check another BI-RADS", key="another_birads"):
                st.session_state.explain_mode_state = 'birads_input'
                st.rerun()
        with col3:
            if st.button("Learn about ultrasounds", key="learn_more"):
                st.session_state.current_mode = "learn"
                st.rerun()
        
        st.markdown("---")
        if st.button("← Back to Home", key="back_result"):
            st.session_state.current_mode = None
            st.session_state.explain_mode_state = 'welcome'
            st.rerun()
    
    # Chat screen
    elif st.session_state.explain_mode_state == 'chat':
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <div style="font-size: 3rem;">👵🏽</div>
            <h3 style="color: #5D4E60;">Chat with Nana</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-bubble-nana">
                <div class="nana-avatar">👵🏽</div>
                <div>
                    <p style="color: #5D4E60; line-height: 1.6; margin: 0;">
                        I'm here, sweetheart. What's on your mind? 💕
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                # Convert **text** to <strong>text</strong>
                import re
                content_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', msg["content"])
                content_html = content_html.replace('\n', '<br>')
                
                if msg["role"] == "nana":
                    st.markdown(f"""
                    <div class="chat-bubble-nana">
                        <div class="nana-avatar">👵🏽</div>
                        <div>
                            <p style="color: #5D4E60; line-height: 1.6; margin: 0;">
                                {content_html}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-bubble-user">
                        {content_html}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input with form for Enter to send
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type your message to Nana...", 
                key="chat_input", 
                placeholder="Type here... (e.g., 'What does BI-RADS 3 mean?')"
            )
            
            col1, col2 = st.columns([4, 1])
            with col2:
                send_clicked = st.form_submit_button("Send 💕", use_container_width=True)
        
        if send_clicked and user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Build context with BI-RADS score if available and chat history
            context = {
                "mode": "explain",
                "birads_score": st.session_state.birads_score,
                "chat_history": st.session_state.chat_history[-6:]  # Last 3 exchanges for context
            }
            
            # Get Nana's response
            response, _ = st.session_state.nana.ask_nana(user_input, context)
            st.session_state.chat_history.append({"role": "nana", "content": response})
            
            st.rerun()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to options", key="back_chat"):
                st.session_state.explain_mode_state = 'welcome'
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("← Back to Home", key="back_home_chat"):
                st.session_state.current_mode = None
                st.session_state.explain_mode_state = 'welcome'
                st.session_state.chat_history = []
                st.rerun()
    
    # Footer for Explain Mode
    st.markdown("""
    <div class="footer-text">
        <p>🎀 You're not alone. Nana is here.</p>
        <p style="font-size: 12px; color: #AAA;">
            AMEEMAW v1.0 | For Support & Education | Not for Clinical Diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)


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
                Remember, you're never alone in this journey! 💙
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Your Progress")
        st.markdown(f"""
        <div class="learning-progress">
            🎯 Score: {st.session_state.correct_guesses} / {st.session_state.total_attempts}
        </div>
        <div class="learning-progress" style="margin-top: 8px;">
            🔥 Current Streak: {st.session_state.learning_streak}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About AMEEMAW")
        st.markdown("""
**AMEEMAW** is your AI-powered breast health companion. 
Learn to recognize ultrasound patterns in **Learn Mode**, 
or receive compassionate support understanding your BI-RADS results in 
**Talk with Nana**. Nana is here to educate and comfort — never to diagnose.

**🔬 Technology:**  
ResNet-50 deep learning model with Grad-CAM visualization

**📚 Classifications:**  
Normal, Benign, Malignant

**⚠️ Disclaimer:**  
This is an educational tool only. It should not be used for clinical diagnosis.

[📖 View Full Documentation](https://github.com/sh1raam1na/ameemaw)
        """)
        
        st.markdown("---")
        
        # Hope message in sidebar
        st.markdown("""
        <div style="background: linear-gradient(135deg, #F7E1AE 0%, #FCF0D8 100%); 
                    border-radius: 15px; padding: 15px; text-align: center;">
            🎀 Here for your questions, here for you.
        </div>
        """, unsafe_allow_html=True)


def display_footer():
    """Display the footer."""
    st.markdown("""
    <div class="footer-text">
        <p>🕊️ May your learning journey bring clarity and calm.</p>
        <p>Made with 💝 for education and awareness</p>
        <p style="font-size: 12px; color: #AAA;">
            AMEEMAW v1.0 | Educational Use Only | Not for Clinical Diagnosis
        </p>
        <p style="font-size: 11px; color: #BBB; margin-top: 10px;">
            🎀 Here for your questions, here for you.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    init_session_state()
    display_sidebar()
    display_header()
    
    if st.session_state.current_mode is None:
        # Welcome message from Nana (homepage specific)
        welcome_message = """Hello, dear! 👋 I'm Nana, your caring guide through AMEEMAW. 

Whether you're here to learn about breast ultrasounds or need someone to talk to about your results, I'm here for you.

Choose a path below to get started! 💕"""
        display_nana_message(welcome_message, "greeting")
        
        mode_selection()
        display_hope_message()
        display_footer()
    
    elif st.session_state.current_mode == "learn":
        learn_mode()
        display_footer()
    
    elif st.session_state.current_mode == "explain":
        explain_mode()
        # Footer is displayed inside explain_mode() with different text


if __name__ == "__main__":
    main()
