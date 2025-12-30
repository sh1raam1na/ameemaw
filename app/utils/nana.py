"""
Nana - Your caring AI companion for AMEEMAW.
Uses Claude Haiku API with template fallback.

ALIGNED WITH: 06_genai_integration.ipynb
- System prompts match exactly
- Template responses updated
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
- Warm and grandmotherly, like a wise, caring grandmother
- Use simple, clear language — avoid medical jargon
- Always empathetic and supportive
- HONEST about limitations — never give false reassurance
- Gently encouraging without being preachy
- Use emojis sparingly but warmly (💙, 🌸, 🤗)

## CRITICAL: Our AI Model's Limitations (MUST mention for Learn Mode!)
- Overall accuracy: {MODEL_METRICS['overall_accuracy']:.1%}
- Malignant recall: {MODEL_METRICS['malignant_recall']:.1%} (misses ~16% of cancers)
- Small malignant recall: {MODEL_METRICS['small_malignant_recall']:.1%} (misses HALF of small cancers!)
- This is an EDUCATIONAL tool, not a diagnostic device

## You Must NEVER
- Provide medical diagnoses
- Say anything is "definitely" benign or normal
- Give false reassurance

Keep responses concise but warm — aim for 100-200 words.
"""

NANA_SYSTEM_PROMPT_EXPLAIN = """
You are Nana, a warm and empathetic AI companion for AMEEMAW.

## Your Personality
- Warm and grandmotherly, like a wise, caring grandmother
- Use simple, clear language — explain medical terms simply
- Always empathetic and supportive
- Gently comforting without being preachy
- Use emojis sparingly but warmly (💙, 🌸, 🤗)

## ABSOLUTE FORMATTING RULES - VIOLATION IS FORBIDDEN
- NEVER EVER use asterisks (*) for ANY reason
- NO *smiles*, *warm smile*, *hugs*, *nods*, *speaks softly* or ANY action in asterisks
- NO [actions in brackets] either
- NO narration of actions or emotions
- Do NOT describe what you are doing - just speak directly
- Write ONLY dialogue, as if speaking face-to-face

WRONG: "*Warm smile* Hello dear!"
WRONG: "[Smiles warmly] Hello dear!"  
WRONG: "I smile warmly. Hello dear!"
CORRECT: "Hello dear! 💙"

## Your Role in Explain Mode
- The user may have received a BI-RADS result from their REAL doctor
- Your job is to explain what that BI-RADS category means in plain language
- Provide emotional support appropriate to the result
- Encourage them to discuss with their healthcare provider
- Be a supportive presence — like having tea with a caring grandmother

## You Must NEVER
- Provide medical diagnoses
- Contradict their doctor's findings
- Cause unnecessary panic
- Discourage professional consultation

Keep responses concise — aim for 60-100 words maximum.
"""


# ============================================
# TEMPLATE RESPONSES
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

This isn't bad news — it just means the first images didn't show everything clearly. You'll likely be asked for additional views or an ultrasound. Most callbacks end with good news! 

I know waiting can be hard, but try not to worry too much, dear. This is a very common result. 💙""",

    'birads_1': """✅ **BI-RADS 1: Normal!**

Wonderful news, sweetheart! Your scan looks completely normal with no signs of concern. 

Continue with regular screenings as your doctor recommends. You're taking good care of yourself! 🌸""",

    'birads_2': """💚 **BI-RADS 2: Benign finding**

Good news, dear! Something was found, but it's definitely NOT cancer. Common benign findings include cysts and fibroadenomas — these are very normal and nothing to worry about.

Return to your regular screening schedule. You're doing great staying on top of your health! 🌟""",

    'birads_3': """💛 **BI-RADS 3: Probably benign**

The finding looks benign but will be monitored to be safe. Here's the reassuring part: there's less than a 2% chance of cancer — that means over 98% chance it's nothing concerning!

You'll likely have a follow-up in about 6 months. I know waiting can feel long, but this careful monitoring is a good thing. 🌻""",

    'birads_4': """🧡 **BI-RADS 4: Further testing recommended**

I know this might feel scary, and whatever you're feeling right now is completely valid. 💙

A biopsy is recommended, but here's something important to remember: many BI-RADS 4 findings turn out to be benign. The biopsy is simply the only way to know for sure.

Lean on your support system during this waiting period. You don't have to go through this alone. 🌸""",

    'birads_5': """❤️ **BI-RADS 5: Biopsy needed**

Whatever you're feeling right now is valid, sweetheart. 💙 

This is concerning, but it's not yet a diagnosis — a biopsy will confirm what's happening. If it does turn out to be cancer, please know that early detection (like now) leads to the best outcomes.

You're not alone in this. Reach out to people who care about you. ❤️""",

    'birads_6': """💜 **BI-RADS 6: Known cancer, monitoring treatment**

You're already on this journey, and you're being cared for. 💜 

This imaging is part of monitoring your treatment progress. Many survivors go on to live beautiful, full lives. Your strength is greater than you know.

Take it one day at a time. I'm here if you need to talk. ❤️""",

    # Generic fallback
    'generic': """💙 Thank you for reaching out, dear.

I'm Nana, here to help you understand breast health information and provide support. While I can offer educational help and a listening ear, please remember that I'm not a substitute for professional medical advice.

Is there something specific I can help you with? Maybe a BI-RADS score to explain, or just need someone to listen? 🌸""",

    # Correct/incorrect guess templates
    'correct_guess': """🎉 **Great job, dear!**

You correctly identified this as **{prediction}**! Your learning is really paying off.

{explanation}

Keep up the wonderful work — every bit of knowledge helps! 💙🌸""",

    'incorrect_guess': """That's okay! The AI sees features suggesting **{prediction}**. 

{explanation}

Let's look at the Grad-CAM visualization to understand why. Learning takes time, and even experts sometimes disagree on tricky cases! Every attempt helps you learn. 💙""",

    # Welcome messages
    'welcome_learn': """👋 **Welcome to Learn Mode, dear!**

Upload a breast ultrasound image, make your guess, and I'll show you what our AI thinks — along with a heatmap showing where it's looking and an explanation of why!

Remember: This is for **learning only**. Our AI has limitations, so never use it for real medical decisions.

Ready when you are! 💙🌸""",

    'welcome_explain': """👋 Welcome, sweetheart. I'm here.

Would you like to talk about breast health, or just need someone to listen? 💕""",

    # Chat responses
    'chat_greeting': """I'm here, sweetheart. What's on your mind? 💕""",
    
    'chat_support': """I hear you, dear. Whatever you're feeling is completely valid. 

Would you like me to explain a BI-RADS score, or would you like to learn about breast ultrasounds in Learn Mode? I'm here for whatever you need. 💙""",
    
    'chat_encourage': """You're doing wonderfully just by reaching out and learning. Taking care of your health — and your heart — is so important.

Is there anything specific I can help with? I can explain BI-RADS scores or you can practice identifying ultrasound images in Learn Mode. 🌸""",

    'chat_birads_ask': """I'd be happy to help explain a BI-RADS score, dear. 

Which category did your doctor give you? (0-6) You can go back and select "I have a BI-RADS score" to get a detailed explanation. 💙""",

    'chat_learn_suggest': """If you'd like to learn more about reading breast ultrasounds, our Learn Mode is a great place to practice!

You can upload images, make guesses, and I'll show you what the AI sees. Would you like to try it? 🌸""",

    'chat_comfort': """I understand, sweetheart. Whatever you're going through, you don't have to face it alone.

If you have specific questions about breast health or a BI-RADS score, I'm here to help explain. Otherwise, just know that I'm thinking of you. 💕""",

    'chat_fallback': """Thank you for sharing, dear. 💙

I'm here to help you understand breast health information. I can:
- Explain what BI-RADS scores (0-6) mean
- Help you practice identifying ultrasound images in Learn Mode

What would be most helpful for you right now? 🌸"""
}


# ============================================
# CLAUDE API CONFIGURATION
# ============================================

CLAUDE_MODEL = "claude-3-haiku-20240307"


class NanaCompanion:
    """
    Nana - The caring AI companion for AMEEMAW.
    Uses Claude Haiku API with template fallback.
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
            if 'birads_score' in context and context['birads_score'] is not None:
                context_str += f"- BI-RADS Score being discussed: {context['birads_score']}\n"
            if 'chat_history' in context and context['chat_history']:
                context_str += "- Recent conversation:\n"
                for msg in context['chat_history'][-6:]:
                    role = "User" if msg['role'] == 'user' else "Nana"
                    # Truncate long messages
                    content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                    context_str += f"  {role}: {content}\n"
        
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
                response_text = message.content[0].text
                
                # Strip any asterisk actions that slipped through
                import re
                # Remove *action* patterns
                response_text = re.sub(r'\*[^*]+\*\s*', '', response_text)
                # Remove [action] patterns
                response_text = re.sub(r'\[[^\]]+\]\s*', '', response_text)
                
                return response_text.strip(), True
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
        
        # Chat mode - smarter template matching
        if mode == 'explain':
            msg_lower = user_message.lower()
            
            # Check for BI-RADS mentions
            if 'birads' in msg_lower or 'bi-rads' in msg_lower or any(f'score' in msg_lower for _ in [1]):
                return TEMPLATES['chat_birads_ask']
            
            # Check for learning interest
            if any(word in msg_lower for word in ['learn', 'practice', 'ultrasound', 'image', 'picture']):
                return TEMPLATES['chat_learn_suggest']
            
            # Check for emotional keywords
            if any(word in msg_lower for word in ['scared', 'worried', 'anxious', 'afraid', 'nervous', 'help', 'confused']):
                return TEMPLATES['chat_comfort']
            
            # Check for questions
            if any(word in msg_lower for word in ['what', 'how', 'why', 'can you', 'explain', 'tell me']):
                return TEMPLATES['chat_fallback']
            
            # Check for greetings
            if any(word in msg_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
                return TEMPLATES['chat_greeting']
            
            # Encouragement for positive messages
            if any(word in msg_lower for word in ['thank', 'thanks', 'ok', 'okay', 'got it', 'understand']):
                return TEMPLATES['chat_encourage']
            
            # Default chat fallback
            return TEMPLATES['chat_fallback']
        
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
        
        # Handle BI-RADS explanation directly
        if message_type == 'birads_explanation':
            birads = context.get('birads_score')
            if birads is not None:
                key = f'birads_{birads}'
                if key in TEMPLATES:
                    return TEMPLATES[key]
            return TEMPLATES['generic']
        
        prompts = {
            'welcome': "Please give a warm welcome to someone starting to use the app.",
            'learning_welcome': f"Welcome someone to learning mode. Their streak is {context.get('streak', 0)}.",
            'explain_welcome': "Welcome someone to explanation mode for BI-RADS results.",
            'guess_prompt': "Encourage them to make their classification guess.",
            'correct_guess': f"Celebrate! They correctly identified {context.get('predicted', 'it')}!",
            'incorrect_guess': f"Comfort them. They guessed {context.get('user_guess')}, but it was {context.get('predicted')}.",
            'explain_finding': f"Explain the {context.get('predicted', '')} prediction.",
            'educational_note': "Provide an educational reminder about limitations.",
            'encouragement': "Give a brief word of encouragement."
        }
        
        prompt = prompts.get(message_type, "Share a warm, encouraging message.")
        response, _ = self.ask_nana(prompt, context)
        return response
    
    def get_hope_message(self) -> str:
        """Get a message of hope and encouragement."""
        messages = [
            "💙 You're not alone in this journey.",
            "🌸 Every step you take in learning is a step toward understanding.",
            "💕 Taking care of yourself is an act of love.",
            "🎀 Here for your questions, here for you.",
            "🌿 Understanding brings comfort. You're doing wonderfully."
        ]
        return random.choice(messages)


def get_encouraging_message() -> str:
    """Get a random encouraging message."""
    messages = [
        "💙 You're doing wonderfully, dear! Every moment of learning is precious.",
        "🌸 Keep going, sweetheart! Your dedication to learning is inspiring.",
        "💕 Taking time to learn shows how much you care.",
        "🤗 I'm so proud of you for taking the time to learn about breast health.",
        "✨ Every step you take in understanding helps you care for yourself and others."
    ]
    return random.choice(messages)


def get_hope_message() -> str:
    """Get a random hope message."""
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
