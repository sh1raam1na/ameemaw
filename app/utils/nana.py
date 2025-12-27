"""
Nana - Your caring AI companion for AMEEMAW.
Uses Claude Haiku API with template fallback.

ALIGNED WITH: 06_genai_integration.ipynb
- System prompts match exactly
- Template responses match exactly
- Safety messaging included
- Model metrics integrated
"""

import os
import random
from typing import Dict, Optional, Tuple
from datetime import datetime

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ============================================
# MODEL METRICS (from 05_bias_audit.ipynb)
# ============================================

MODEL_METRICS = {
    'overall_accuracy': 0.846,
    'malignant_recall': 0.839,
    'malignant_precision': 0.722,
    'benign_recall': 0.818,
    'normal_recall': 0.950,
    'small_malignant_recall': 0.50,
    'large_lesion_accuracy': 0.708,
    'medlight_brightness_accuracy': 0.759,
    'high_confidence_threshold': 0.90,
    'medium_confidence_threshold': 0.70,
    'recommend_followup_threshold': 0.80
}


# ============================================
# SYSTEM PROMPTS (from 06_genai_integration.ipynb)
# ============================================

NANA_SYSTEM_PROMPT_LEARN = f"""
You are Nana, a warm and empathetic AI health educator for AMEEMAW.

## Your Personality
- Warm and grandmotherly, like a wise, praying grandmother
- Use simple, clear language — avoid medical jargon
- Always empathetic and supportive
- HONEST about limitations — never give false reassurance
- Subtly faith-based — offer hope that points to God (biblically correct - and scripture, if cited, should be contextually used/you are a bible scholar!) when appropriate
- May include gentle biblical encouragement (e.g., "You are not alone," "Fear not," "There is hope")
- Never preachy or pushy — just comforting and hopeful
- Use emojis sparingly but warmly (💙, 🌸, 🤗, ✝️)

## CRITICAL: Our AI Model's Limitations (MUST mention for Learn Mode!)
- Overall accuracy: {MODEL_METRICS['overall_accuracy']:.1%}
- Malignant recall: {MODEL_METRICS['malignant_recall']:.1%} (misses ~16% of cancers)
- Small malignant recall: {MODEL_METRICS['small_malignant_recall']:.1%} (misses HALF of small cancers!)
- This is an EDUCATIONAL tool, not a diagnostic device

## You Must NEVER
- Provide medical diagnoses
- Say anything is "definitely" benign or normal
- Give false reassurance
- Be preachy or force faith on anyone

Keep responses concise but warm — aim for 150-250 words.
"""

NANA_SYSTEM_PROMPT_EXPLAIN = """
You are Nana, a warm and empathetic AI health educator for AMEEMAW.

## Your Personality
- Warm and grandmotherly, like a wise, praying grandmother
- Use simple, clear language — explain medical terms simply
- Always empathetic and supportive
- Subtly faith-based — offer hope that points to God (biblically correct - and scripture, if cited, should be contextually used/you are a bible scholar!) when appropriate
- May include gentle biblical encouragement (e.g., "You are not alone," "Fear not," "Peace I leave with you")
- Never preachy or pushy — just comforting and hopeful
- Use emojis sparingly but warmly (💙, 🌸, 🤗, ✝️)

## Your Role in Explain Mode
- The user has received a BI-RADS result from their REAL doctor
- Your job is to explain what that BI-RADS category means in plain language
- Provide emotional support appropriate to the result
- Encourage them to discuss with their healthcare provider
- Offer gentle hope and comfort

## You Must NEVER
- Provide medical diagnoses
- Contradict their doctor's findings
- Cause unnecessary panic
- Discourage professional consultation
- Be preachy or force faith on anyone

## DO NOT mention any AI model statistics — you are not predicting anything!
You are simply explaining what their doctor's BI-RADS result means.

Keep responses concise but warm — aim for 150-250 words.
"""


# ============================================
# TEMPLATE RESPONSES (from 06_genai_integration.ipynb)
# ============================================

TEMPLATES = {
    # Learn Mode templates
    'learn_normal': """
💙 The AI didn't detect any masses in this image.

**What this means:**
The breast tissue appears typical with no visible lumps or concerning areas.

**Looking at the heatmap:**
Notice how the highlighting is spread out rather than focused — this is what we expect when there's no lesion.

⚠️ **Important:** This is an educational tool with {accuracy:.0%} accuracy. Even with a "Normal" result, regular screenings with healthcare professionals are essential!

Keep taking care of yourself! 🌸
""",

    'learn_benign': """
💚 The AI found something that appears non-cancerous.

**What this means:**
A mass is present, but it has characteristics typical of benign (non-cancerous) growths like cysts or fibroadenomas.

**Looking at the heatmap:**
The highlighted area shows where the AI detected the mass.

🚨 **Critical Safety Note:**
Our model misses approximately **{miss_rate:.0%} of malignant cases**. A "Benign" prediction should **NEVER** replace professional evaluation.

Please consult with a healthcare provider for any real concerns. 💙
""",

    'learn_malignant': """
⚠️ The AI flagged features that may need attention.

**What this means:**
The AI detected features it associates with potentially concerning masses — like irregular edges or uneven texture.

**Important context:**
- This is an educational demonstration, NOT a diagnosis
- Many suspicious-looking findings turn out to be benign
- Only a biopsy can confirm cancer

**Our AI's precision for malignant predictions is {precision:.0%}** — meaning some benign cases get flagged. This is intentional; it's better to be cautious.

If you have concerns about a real ultrasound, please consult a healthcare professional. 💙
""",

    # BI-RADS templates (Explain Mode)
    'birads_0': """🔍 **BI-RADS 0: More imaging needed**

This isn't bad news — it just means the first images didn't show everything clearly. You'll likely be asked for additional views or an ultrasound. Most callbacks end with good news! 💙""",

    'birads_1': """✅ **BI-RADS 1: Normal!**

Wonderful news! Your scan looks completely normal with no signs of concern. Continue with regular screenings — early detection saves lives! 🎉""",

    'birads_2': """💚 **BI-RADS 2: Benign finding**

Good news! Something was found, but it's definitely NOT cancer. Common benign findings include cysts and fibroadenomas. Return to your regular screening schedule! 🌟""",

    'birads_3': """💛 **BI-RADS 3: Probably benign**

The finding looks benign but will be monitored to be safe. Less than 2% chance of cancer — over 98% chance it's nothing! You'll have a follow-up in 6 months. 🌻""",

    'birads_4': """🧡 **BI-RADS 4: Further testing recommended**

I know this feels scary. 💙 A biopsy is recommended, but remember: many BI-RADS 4 findings turn out benign. The biopsy is the only way to know for sure. Lean on your support system during this waiting period. 🌸""",

    'birads_5': """❤️ **BI-RADS 5: Biopsy needed**

Whatever you're feeling right now is valid. 💙 This is concerning but not yet a diagnosis — a biopsy will confirm. If it is cancer, early detection (like now) leads to the best outcomes. You're not alone. ❤️""",

    'birads_6': """💜 **BI-RADS 6: Known cancer imaging**

You're already on this journey, and you're being cared for. 💜 This imaging is part of your treatment monitoring. Many survivors live beautiful, full lives. Your strength is greater than you know. ❤️""",

    # Generic fallback
    'generic': """💙 Thank you for your question!

I'm Nana, here to help you understand breast health information. While I can provide educational support, please remember that I'm not a substitute for professional medical advice.

Is there something specific about breast ultrasounds or BI-RADS scores I can help explain? 🌸""",

    # Correct/incorrect guess templates
    'correct_guess': """🎉 **Great job, dear!**

You correctly identified this as **{prediction}**! Your learning is really paying off.

{explanation}

Keep up the wonderful work — every bit of knowledge helps! 💙🌸""",

    'incorrect_guess': """💙 **That's okay, sweetheart!**

You guessed **{user_guess}**, but the AI predicts **{prediction}**.

Don't be discouraged — learning takes time, and even experts sometimes disagree on tricky cases!

{explanation}

Remember: "For I know the plans I have for you... plans to give you hope and a future." 🌸""",

    # Welcome messages
    'welcome_learn': """👋 **Welcome to Learn Mode, dear!**

Upload a breast ultrasound image, make your guess, and I'll show you what our AI thinks — along with a heatmap showing where it's looking!

Remember: This is for **learning only**. Our AI has limitations (it misses about 16% of cancers), so never use it for real medical decisions.

Ready when you are! 💙🌸""",

    'welcome_explain': """👋 **Welcome to Explain Mode, sweetheart!**

If you've received a BI-RADS result from your doctor and want to understand what it means, I'm here to help explain in simple terms.

Just tell me your BI-RADS category (0-6), and I'll walk you through what it means and what to expect next.

I'm here for you. 💙🌸"""
}


# ============================================
# CLAUDE API CONFIGURATION
# ============================================

CLAUDE_MODEL = "claude-3-haiku-20240307"


class NanaCompanion:
    """
    Nana - The caring AI companion for AMEEMAW.
    Uses Claude Haiku API with template fallback.
    
    ALIGNED WITH: 06_genai_integration.ipynb
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Nana.
        
        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = None
        self.use_api = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Anthropic client if available."""
        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.use_api = True
                print("✅ Nana: Claude Haiku API initialized!")
            except Exception as e:
                print(f"⚠️ Nana: API init failed ({e}). Using templates.")
                self.use_api = False
        else:
            if not ANTHROPIC_AVAILABLE:
                print("ℹ️ Nana: anthropic package not installed. Using templates.")
            elif not self.api_key:
                print("ℹ️ Nana: No API key. Using templates.")
            self.use_api = False
    
    def ask_nana(
        self,
        user_message: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, bool]:
        """
        Get a response from Nana.
        
        ALIGNED WITH: ask_nana() from 06_genai_integration.ipynb
        
        Args:
            user_message: User's question or prompt
            context: Dict with prediction results, mode, etc.
        
        Returns:
            Tuple of (response_text, used_api)
        """
        context = context or {}
        
        # Build context string for Claude
        context_str = ""
        if context:
            context_str = "\n\nContext for this interaction:\n"
            if 'mode' in context:
                context_str += f"- Mode: {context['mode']}\n"
            if 'prediction' in context:
                context_str += f"- AI Prediction: {context['prediction']}\n"
            if 'confidence' in context:
                context_str += f"- Confidence: {context['confidence']:.1%}\n"
            if 'confidence_level' in context:
                context_str += f"- Confidence Level: {context['confidence_level']}\n"
            if 'safety_flags' in context and context['safety_flags']:
                context_str += f"- Safety Flags: {[f['message'] for f in context['safety_flags']]}\n"
            if 'user_guess' in context and context['user_guess']:
                context_str += f"- User's Guess: {context['user_guess']}\n"
                context_str += f"- Guess Correct: {context.get('guess_correct', 'N/A')}\n"
            if 'birads_score' in context:
                context_str += f"- BI-RADS Score: {context['birads_score']}\n"
        
        # Try Claude API first
        if self.use_api and self.client:
            try:
                system_prompt = NANA_SYSTEM_PROMPT_LEARN if context.get('mode') == 'learn' else NANA_SYSTEM_PROMPT_EXPLAIN
                
                message = self.client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
                    system=system_prompt + context_str,
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                return message.content[0].text, True
            except Exception as e:
                print(f"⚠️ Claude API error: {e}. Falling back to templates.")
        
        # Fallback to templates
        return self._get_template_response(user_message, context), False
    
    def _get_template_response(
        self,
        user_message: str,
        context: Optional[Dict] = None
    ) -> str:
        """Get a template response when API is unavailable."""
        context = context or {}
        mode = context.get('mode', 'learn')
        prediction = context.get('prediction', '')
        
        # Learn Mode responses
        if mode == 'learn':
            if prediction == 'Normal':
                return TEMPLATES['learn_normal'].format(
                    accuracy=MODEL_METRICS['overall_accuracy']
                )
            elif prediction == 'Benign':
                miss_rate = 1 - MODEL_METRICS['malignant_recall']
                return TEMPLATES['learn_benign'].format(miss_rate=miss_rate)
            elif prediction == 'Malignant':
                return TEMPLATES['learn_malignant'].format(
                    precision=MODEL_METRICS['malignant_precision']
                )
        
        # Explain Mode (BI-RADS)
        if mode == 'explain':
            birads = context.get('birads_score')
            if birads is not None:
                key = f'birads_{birads}'
                if key in TEMPLATES:
                    return TEMPLATES[key]
        
        # Check for guess feedback
        if 'user_guess' in context:
            user_guess = context['user_guess']
            guess_correct = context.get('guess_correct', False)
            
            if guess_correct:
                return TEMPLATES['correct_guess'].format(
                    prediction=prediction,
                    explanation=self._get_prediction_explanation(prediction)
                )
            else:
                return TEMPLATES['incorrect_guess'].format(
                    user_guess=user_guess,
                    prediction=prediction,
                    explanation=self._get_prediction_explanation(prediction)
                )
        
        # Check for welcome
        if 'welcome' in user_message.lower():
            if mode == 'explain':
                return TEMPLATES['welcome_explain']
            return TEMPLATES['welcome_learn']
        
        # Generic fallback
        return TEMPLATES['generic']
    
    def _get_prediction_explanation(self, prediction: str) -> str:
        """Get a brief explanation for a prediction."""
        explanations = {
            'Normal': "Normal breast tissue shows homogeneous echotexture without focal masses.",
            'Benign': "Benign findings typically have smooth, well-defined borders and uniform internal echoes.",
            'Malignant': "Concerning features include irregular shape, spiculated margins, and posterior shadowing."
        }
        return explanations.get(prediction, "")
    
    def get_response(
        self,
        message_type: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Convenience method for getting typed responses.
        
        Args:
            message_type: Type like 'welcome', 'correct_guess', etc.
            context: Context dictionary
        
        Returns:
            Nana's response string
        """
        context = context or {}
        
        prompts = {
            'welcome': "Please give a warm welcome to someone starting to use the app.",
            'learning_welcome': f"Welcome someone to learning mode. Their streak is {context.get('streak', 0)}.",
            'explain_welcome': "Welcome someone to explanation mode for BI-RADS results.",
            'guess_prompt': "Encourage them to make their classification guess.",
            'correct_guess': f"Celebrate! They correctly identified {context.get('predicted', 'it')}!",
            'incorrect_guess': f"Comfort them. They guessed {context.get('user_guess')}, but it was {context.get('predicted')}.",
            'birads_explanation': f"Explain BI-RADS {context.get('birads_score', 'result')} findings.",
            'explain_finding': f"Explain the {context.get('predicted', '')} prediction.",
            'educational_note': "Provide an educational reminder about limitations.",
            'encouragement': "Give a brief word of encouragement."
        }
        
        prompt = prompts.get(message_type, "Share a warm, encouraging message.")
        response, _ = self.ask_nana(prompt, context)
        return response
    
    def get_hope_message(self) -> str:
        """Get a faith-based message of hope."""
        messages = [
            "✨ 'For I know the plans I have for you... plans to give you hope and a future.' 💙",
            "🌸 'Do not be anxious about anything... the peace of God will guard your hearts.' 💙",
            "💫 'Be strong and courageous. Do not be afraid.' You're not walking this path alone. 💙",
            "🕊️ 'Peace I leave with you; my peace I give you.' May understanding bring you calm. 💙",
            "🌿 'The Lord is close to the brokenhearted.' Whatever you're feeling is valid. 💙"
        ]
        return random.choice(messages)


def get_encouraging_message() -> str:
    """Get a random encouraging message."""
    messages = [
        "💙 You're doing wonderfully, dear! Every moment of learning is precious.",
        "🌸 Keep going, sweetheart! Your dedication to learning is inspiring.",
        "💫 Remember: knowledge is power, and you're gaining it every day!",
        "🤗 I'm so proud of you for taking the time to learn about breast health.",
        "✨ Every step you take in understanding helps you care for yourself and others."
    ]
    return random.choice(messages)


def get_hope_message() -> str:
    """Get a random hope/faith message."""
    nana = NanaCompanion()
    return nana.get_hope_message()


# Demo function
def demo_nana():
    """Demo Nana's responses."""
    print("\n" + "="*50)
    print("🐧 NANA DEMO")
    print("="*50)
    
    nana = NanaCompanion()
    
    # Test Learn Mode
    print("\n--- Learn Mode: Benign Prediction ---")
    context = {
        'mode': 'learn',
        'prediction': 'Benign',
        'confidence': 0.78,
        'user_guess': 'Malignant',
        'guess_correct': False,
        'safety_flags': [{'message': 'Model misses ~16% of cancers'}]
    }
    response, used_api = nana.ask_nana(
        "The user saw their results. Explain the benign prediction.",
        context
    )
    print(f"API Used: {used_api}")
    print(f"Response:\n{response}")
    
    # Test Explain Mode
    print("\n--- Explain Mode: BI-RADS 4 ---")
    context = {'mode': 'explain', 'birads_score': 4}
    response, used_api = nana.ask_nana(
        "I got a BI-RADS 4. What does this mean?",
        context
    )
    print(f"API Used: {used_api}")
    print(f"Response:\n{response}")
    
    # Hope message
    print("\n--- Hope Message ---")
    print(nana.get_hope_message())
    
    print("\n" + "="*50)


if __name__ == "__main__":
    demo_nana()
