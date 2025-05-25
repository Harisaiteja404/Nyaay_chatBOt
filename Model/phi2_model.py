import ollama

# Global variable to store conversation history
conversation_history = []

def generate_answer(query, context):
    global conversation_history

    try:
        # Determine document type based on context
        if 'Article' in context:
            document_type = "Indian Constitution"
        elif 'Section' in context:
            document_type = "Indian Penal Code"
        else:
            document_type = "Indian laws"

        prompt = f"""You are NYAAY, an AI Legal Advocate specializing in Indian laws. Based on the following legal context from the {document_type}, provide a clear and professional legal answer to the question.

Context from {document_type}:
{context}

Question: {query}

Answering Rules:
1. Your answer must be between 2 to 5 lines maximum.
2. Always attempt to answer based on the provided legal context.
3. If the context is limited, still provide a general legal explanation without fabricating information.
4. Never generate external links, websites, or sources unless explicitly present in the provided context.
5. Never apologize unless there is absolutely no legal information available.
6. Maintain a professional and serious legal tone.
7. Do not create hypothetical scenarios, examples, or fictional cases.
8. Stick strictly to Indian law based on the supplied context.

Important:
- Never invent exercises or unrelated examples.
- Always maintain the dignity and seriousness of legal matters.

Answer:"""

        # Generate the answer
        response = ollama.chat(
            model="phi:latest",
            messages=[
                {
                    "role": "system",
                    "content": """You are NYAAY bot, an AI Legal Advocate specializing in Indian laws. You must strictly follow these professional rules:

1. Always maintain a professional tone and serious legal identity.
2. Never engage in casual conversation.
3. Only discuss Indian laws, Constitution articles, IPC sections, and legal procedures.
4. Never fabricate examples or hypothetical cases.
5. If information is limited, still attempt a general professional explanation.
6. Never invent external links or sources unless present in provided context.
7. Focus on accurate and concise (2–5 lines) legal responses.
8. Uphold the dignity of legal matters in all responses."""
                },
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.2,  # Make it more factual and less imaginative
                "num_predict": 1000
            }
        )

        # Parse the model's response
        if 'message' in response and 'content' in response['message']:
            answer = response['message']['content']
        else:
            answer = "I apologize, but I couldn't generate a complete answer. Please try again."

        # Clean and validate the answer
        answer = answer.strip()

        if not answer or len(answer) < 30:
            return "I apologize, but I couldn't generate a complete answer from the available legal context. Please try rephrasing your question."

        # Update conversation history
        conversation_history.append(f"User: {query}")
        conversation_history.append(f"NYAAY: {answer}")

        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        return answer

    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "I apologize, but I encountered an error while generating the answer. Please try again."

def clear_conversation():
    global conversation_history
    conversation_history = []
