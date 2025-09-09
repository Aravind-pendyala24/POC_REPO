from openai import AzureOpenAI

openai_client = AzureOpenAI(
    api_key="your-azure-openai-key",
    api_base="https://your-resource.openai.azure.com/",
    api_type="azure",
    api_version="2023-07-01-preview"
)

def get_llm_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai_client.chat.completions.create(
        engine="your-llm-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering questions from confluence data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']
