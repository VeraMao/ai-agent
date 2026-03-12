import json
import boto3
import os
import requests
from bs4 import BeautifulSoup

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
GENIUS_API_TOKEN = os.environ['GENIUS_API_TOKEN']
bucket_name = 'genius-bucket-654654514107'
table_name = 'genius-table'

def lambda_handler(event, context):
    table = dynamodb.Table(table_name)

    for record in event['Records']:
        body = json.loads(record['body'])
        title = body['title']
        artist = body['artist']
        track_id = body['track_id']

        headers = {'Authorization': f'Bearer {GENIUS_API_TOKEN}'}
        search_url = f"https://api.genius.com/search?q={title} {artist}"
        response = requests.get(search_url, headers=headers).json()

        try:
            song_info = response['response']['hits'][0]['result']
            song_url = song_info['url']

            # scrape lyrics
            lyrics_page = requests.get(song_url)
            soup = BeautifulSoup(lyrics_page.text, 'html.parser')

            # NEW: Look for all data-lyrics-container divs
            lyrics_divs = soup.find_all("div", {"data-lyrics-container": "true"})
            if lyrics_divs:
                lyrics = "\n".join(div.get_text(separator="\n").strip() for div in lyrics_divs)
            else:
                lyrics = "Lyrics not found."

        except (IndexError, AttributeError):
            song_url = "Not found"
            lyrics = "Lyrics not found."

        # Save to S3
        s3_key = f"lyrics/{track_id}.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps({
                'track_id': track_id,
                'title': title,
                'artist': artist,
                'url': song_url,
                'lyrics': lyrics
            })
        )
        print(f"Saved to S3: {s3_key}")

        # Save to DynamoDB
        table.put_item(
            Item={
                'user_id': track_id,
                'title': title,
                'artist': artist,
                'url': song_url,
                'lyrics': lyrics
            }
        )

    return {
        'statusCode': 200,
        'body': json.dumps('Processed successfully')
    }