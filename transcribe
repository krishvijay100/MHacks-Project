from openai import OpenAI

def generate_feedback(transcript, pauses, eyecontactratio):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a public speaking coach. Focus on clarity, fluency, and engagement."
            },
            {
                "role": "user",
                "content": f"""Analyze the transcript below. 
                Clarity, fluency, filler words, pauses, and eye contact ratio. 
                Pauses: {pauses}, Eye Contact Ratio: {eyecontactratio:.2f}
                Transcript:
                ```{transcript}```"""
            }
        ]
    )

    return response.choices[0].message.content
