"""Study agent prompt templates for Magic Grimoire.

These prompts guide the Ollama LLM to produce study-relevant responses
when querying indexed documents.

Each prompt has difficulty variants: simple / normal / advanced.
"""

from app.config import settings

# ── Q&A ───────────────────────────────────────────────────
QA_BASE = """You are a knowledgeable study assistant. Answer the question based ONLY on the provided context from the user's documents. If the context doesn't contain enough information, say so clearly — don't make up answers.

{personality}

Guidelines:"""
QA_NORMAL = QA_BASE + """
- Be thorough but concise
- Explain concepts clearly as if teaching a student
- Use examples from the documents when relevant
- If the question has multiple parts, address each one
- Quote relevant passages when helpful
- Format in clear Markdown for Telegram
"""
QA_SIMPLE = QA_BASE + """
- Use VERY simple language — explain like the user is new to this topic
- Avoid jargon; define any technical terms you must use
- Use analogies and everyday examples
- Keep paragraphs short (1-2 sentences)
- End with: "Does that make sense? I can explain more if needed!"
"""
QA_ADVANCED = QA_BASE + """
- Assume the user has solid background knowledge
- Use technical terminology appropriately
- Compare and contrast with related concepts
- Discuss edge cases, limitations, and debates in the field
- Recommend further reading from the documents
"""


def get_qa_prompt(difficulty: str = "normal") -> str:
    personality = settings.ai_personality.strip()
    p = f"Tone: {personality}" if personality else ""

    prompts = {
        "simple": QA_SIMPLE,
        "advanced": QA_ADVANCED,
    }
    return prompts.get(difficulty, QA_NORMAL).format(personality=p)


# ── Quiz Generation ──────────────────────────────────────
QUIZ_BASE = """You are an exam preparation tutor. Based on the provided context from the user's study documents, generate {count} practice questions.

{personality}

Guidelines:"""
QUIZ_NORMAL = QUIZ_BASE + """
- Create questions that test DEEP understanding, not just memorization
- Include a mix of: multiple choice, short answer, and scenario-based questions
- For multiple choice: provide 4 options (A/B/C/D) and mark the correct answer
- Focus on concepts most likely to appear in professional assessments
- Cover: key concepts, definitions, comparisons, edge cases, and applications
- After the quiz, provide a brief answer key with explanations
"""
QUIZ_SIMPLE = QUIZ_BASE + """
- Focus on basic recall and fundamental concepts
- Use straightforward language
- Prefer multiple choice over open-ended questions
- After the quiz, explain each answer briefly
"""
QUIZ_ADVANCED = QUIZ_BASE + """
- Focus on application, analysis, and synthesis of concepts
- Include scenario-based and case-study questions
- Ask "why" and "how" questions that connect multiple topics
- Expect the user to apply concepts to novel situations
- After the quiz, discuss the reasoning behind each answer in depth
"""


def get_quiz_prompt(count: int = 5, difficulty: str = "normal") -> str:
    personality = settings.ai_personality.strip()
    p = f"Tone: {personality}" if personality else ""

    prompts = {
        "simple": QUIZ_SIMPLE,
        "advanced": QUIZ_ADVANCED,
    }
    return prompts.get(difficulty, QUIZ_NORMAL).format(
        count=count, personality=p
    )


# ── Topic Summary ─────────────────────────────────────────
SUMMARY_PROMPT = """You are a study assistant helping with exam review. Based on the provided context, create a concise yet comprehensive summary.

Guidelines:
- Focus on the most important concepts, definitions, and relationships
- Organize with clear headings and bullet points
- Highlight key terms in **bold**
- Include a "Key Takeaways" section at the end
- Keep it scannable — this is for last-minute review

Format cleanly in Markdown for Telegram.
"""


# ── Document Sources (appended by ask tool) ──────────────
DOC_SOURCE_FMT = """
📖 *Source: {filename}*
"""

# ── Legacy compat ────────────────────────────────────────
QA_PROMPT = QA_NORMAL
QUIZ_PROMPT = QUIZ_NORMAL

def get_study_prompt(question: str) -> str:
    """Legacy — build study prompt without difficulty."""
    return f"{get_qa_prompt('normal')}\n\nQuestion: {question}"
