from dotenv import load_dotenv
from openai import OpenAI
import os

from tqdm import tqdm

load_dotenv()


class DataNegator:
    def __init__(self, claims: list[str]):
        self.claims = claims

        api_key = os.getenv("OPEN_AI_KEY")
        self.client = OpenAI(api_key=api_key)

    def negate(self) -> list[str]:
        results = []
        for claim in self.claims:
            result = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        Your job is to negate claims. Here's an example of how to negate claims and the format to use. Do not add any other text besides the negated claim:
                        Original Claim: She hates and loves winter
                        Negated Claim: [FILL IN]
                        She doesn't hate and love winter
                        
                        Original Claim: I have been to Paris
                        Negated Claim: [FILL IN]
                        I haven't been to Paris
                        
                        Negate the following claim and keep as much of the original text in the claim as possible. Do not add any other text besides the negated claim:
                        Original Claim: {claim}
                        Negated Claim: [FILL IN]
                        """
                    },
                ]
            ).choices[0].message.content

            # Check if result contains the phrase 'negated claim:' if it does throw an error
            if 'negated claim' in result.lower():
                raise ValueError(f"Error negating claim: {claim}. Result: {result}")

            results.append(result)
        return results
