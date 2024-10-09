import re

import pandas as pd
import praw
from dotenv import load_dotenv
import os
import requests
import json
import nltk

from praw.models import Subreddit, Comment
from praw.models.comment_forest import CommentForest
from tqdm import tqdm

load_dotenv()


class RedditScraper:
    subreddits: list[Subreddit] = []
    comments: list[Comment] = []

    def __init__(self, limit=1000, claim_limit=1000):
        self.limit = limit
        self.reddit = praw.Reddit(client_id="ML6Izub1j8jaTmA7nEQaiw",
                                  client_secret=os.getenv("REDDIT_KEY"),
                                  user_agent="Reddit Scraper (by poppytom5)"
                                  )
        self.claim_limit = claim_limit

    def find_subreddits(self, query) -> None:
        query = query #+ "&sort=new"
        search_results = self.reddit.subreddit("climateskeptics").search(query, limit=self.limit)
        for i, submission in tqdm(enumerate(search_results)):
            submission.comments.replace_more(limit=0)
            comment_forest: CommentForest = submission.comments
            for comment in comment_forest.list():
                self.comments.append(comment)
            if len(submission.selftext) > 0:
                self.subreddits.append(submission)

        return

    def export_subreddits_to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["title"] = [submission.title for submission in self.subreddits]
        df["score"] = [submission.score for submission in self.subreddits]
        df["id"] = [submission.id for submission in self.subreddits]
        df["url"] = [submission.url for submission in self.subreddits]
        df["comms_num"] = [submission.num_comments for submission in self.subreddits]
        df["created"] = [submission.created for submission in self.subreddits]
        df["body"] = [submission.selftext for submission in self.subreddits]

        for comment in self.comments:
            # Add to df using concat
            df = pd.concat([df, pd.DataFrame({
                "title": "",
                "score": "0",
                "id": comment.id,
                "url": "",
                "comms_num": "",
                "created": "",
                "body": comment.body
            }, index=[comment.id])], ignore_index=True)

        return df

    def get_claims(self, df: pd.DataFrame, claim_column_name: str) -> pd.DataFrame:
        # Convert the dataframe column into a list
        claims = df[claim_column_name].tolist()
        for i, claim in enumerate(claims):
            # Replace all "." in the claim with ", "
            claims[i] = claim.replace(".", ", ")
            claims[i] = claim.replace(";", ", ")
            claims[i] = claim.replace("...", ", ")
            claims[i] = claim.replace("\n", ", ")

        # UNcommet for actual reddit scraping
        # # Join the list into a single string
        # sentences = ". ".join(claims)
        #
        # sentences = re.sub(r'\.\.\d{2}', '', sentences)
        # sentences = re.sub(r'\.\.\d{1}', '', sentences)
        # sentences = re.sub(r'\.\d{2}', '.', sentences)
        # sentences = re.sub(r'\.\d{1}', '.', sentences)
        # # Tokenize the sentences
        # sentences = nltk.sent_tokenize(sentences)
        # # Loop through sentences. If they don't end in a '.' add a '.'
        # for i, sentence in enumerate(sentences):
        #     if sentence[-1] != ".":
        #         sentences[i] = sentence + "."

        sentences = claims
        sentence_batch = ""
        batch_size = 1
        response = pd.DataFrame()
        num_claims_sent = 0
        print(len(sentences))
        for i, sentence in tqdm(enumerate(sentences)):
            sentence_batch += sentence + ". "
            if i % batch_size == 0 or i == len(sentences) - 1:
                response_body = self.__make_request("https://idir.uta.edu/claimbuster/api/v2/score/text/sentences/",
                                                    sentence_batch)
                parsed_response = self.__parse_response(response_body)
                # Concat the parsed response to the response dataframe
                response = pd.concat([response, parsed_response], ignore_index=True)
                sentence_batch = ""
                num_claims_sent += batch_size

            if num_claims_sent > self.claim_limit:
                break

        return response

    def __parse_response(self, response: dict) -> pd.DataFrame:
        df = pd.DataFrame()
        # Get text key in each object in results in response
        df["text"] = [result["text"] for result in response["results"]]
        df["score"] = [result["score"] for result in response["results"]]
        df["index"] = [result["index"] for result in response["results"]]

        return df

    def __make_request(self, url, sentences: str) -> dict:
        api_key = os.getenv("CLAIMBUSTER_KEY")

        # Define the endpoint (url), payload (sentence to be scored), api-key (api-key is sent as an extra header)
        api_endpoint = url
        request_headers = {"x-api-key": api_key}
        payload = {"input_text": sentences}

        # Send the POST request to the API and store the api response
        api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)

        return api_response.json()

    #
    # def get_submissions(self):
    #     submissions = self.reddit.subreddit(self.subreddit).top(limit=self.limit)
    #     return submissions
    #
    # def get_comments(self, submission):
    #     submission.comments.replace_more(limit=0)
    #     return submission.comments.list()
    #
    # def get_submission_data(self, submission):
    #     return {
    #         'id': submission.id,
    #         'title': submission.title,
    #         'score': submission.score,
    #         'num_comments': submission.num_comments,
    #         'created_utc': submission.created_utc,
    #         'selftext': submission.selftext,
    #         'url': submission.url,
    #         'author': submission.author.name,
    #         'subreddit': submission.subreddit.display_name
    #     }
    #
    # def get_comment_data(self, comment):
    #     return {
    #         'id': comment.id,
    #         'body': comment.body,
    #         'score': comment.score,
    #         'created_utc': comment.created_utc,
    #         'author': comment.author.name,
    #         'subreddit': comment.subreddit.display_name
    #     }
    #
    # def get_submission_comments(self, submission):
    #     comments = self.get_comments(submission)
    #     return [self.get_comment_data(comment) for comment in comments]
    #
    # def get_submissions_data(self):
    #     submissions = self.get_submissions()
    #     return [self.get_submission_data(submission) for submission in submissions]
    #
    # def get_submissions_comments_data(self):
    #     submissions = self.get_submissions()
    #     return [(self.get_submission_data(submission), self.get_submission_comments(submission)) for submission in submissions]
