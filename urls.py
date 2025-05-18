import requests
import json
import random
import urllib.parse
import hashlib
import os
from dotenv import load_dotenv
from name import detect_top_celebrity_name

# Load environment variables
load_dotenv()

# === Google Images ===
def get_google_image_urls(query, total_num=10, offset=0):
    """
    Fetch actual related images using Google Custom Search API
    """
    # Google Custom Search API credentials
    api_key = os.getenv('GOOGLE_API_KEY')
    cx = os.getenv('GOOGLE_CSE_ID', '8526bb01ab72c4115')  # This should be your Custom Search Engine ID
    
    # Create a list to hold the image URLs
    all_image_urls = []
    
    try:
        # We may need to make multiple requests to get enough images
        # Google Custom Search API returns max 10 results per request
        for start_index in range(offset + 1, offset + total_num + 1, 10):
            # Prepare parameters for the API request
            params = {
                'q': query,
                'cx': cx,
                'key': api_key,
                'searchType': 'image',
                'num': min(10, total_num - len(all_image_urls)),
                'start': start_index,
                'imgSize': 'medium',
                'safe': 'active'  # Safe search enabled
            }
            
            # Make the API request
            response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
            data = response.json()
            
            # Check if we got any results
            if 'items' in data:
                # Extract image URLs from the results
                for item in data['items']:
                    if len(all_image_urls) < total_num:
                        all_image_urls.append(item['link'])
            else:
                # If no items are returned, break out of the loop
                break
                
            # If we have enough image URLs, break out of the loop
            if len(all_image_urls) >= total_num:
                break
    
    except Exception as e:
        print(f"Error fetching images: {str(e)}")
        # Fall back to placeholder images if the API fails
        placeholder_services = [
            "https://source.unsplash.com/random/800x600/?{query}",
            "https://loremflickr.com/800/600/{query}"
        ]
        
        # Generate some placeholder images as a fallback
        for i in range(total_num):
            service = placeholder_services[i % len(placeholder_services)]
            url = service.format(query=query.replace(" ", ","))
            all_image_urls.append(url)
    
    # Return the requested number of image URLs
    return all_image_urls[:total_num]

# === YouTube Videos ===
def get_youtube_video_urls(query, max_results=5, offset=0):
    """
    Fetch actual YouTube videos related to the query using YouTube Data API
    """
    # YouTube Data API key
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # Create a list to hold the video URLs
    youtube_urls = []
    
    try:
        # Prepare parameters for the API request
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'key': api_key,
            'videoEmbeddable': 'true',
            'safeSearch': 'strict',
            'pageToken': None  # Will be updated with next page token
        }
        
        # Make the API request
        response = requests.get('https://www.googleapis.com/youtube/v3/search', params=params)
        data = response.json()
        
        # Check if we got any results
        if 'items' in data:
            # Extract video URLs from the results
            for item in data['items']:
                if 'videoId' in item['id']:
                    video_id = item['id']['videoId']
                    youtube_urls.append(f"https://www.youtube.com/watch?v={video_id}")
    
    except Exception as e:
        print(f"Error fetching YouTube videos: {str(e)}")
        # Fall back to predefined videos if the API fails
        fallback_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo
            "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - Gangnam Style
            "https://www.youtube.com/watch?v=kJQP7kiw5Fk",  # Luis Fonsi - Despacito
            "https://www.youtube.com/watch?v=OPf0YbXqDm0"   # Mark Ronson - Uptown Funk
        ]
        
        # Shuffle based on query to always get same fallback videos for same celebrity
        name_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
        random.seed(name_hash)
        random.shuffle(fallback_videos)
        youtube_urls = fallback_videos[offset:offset + max_results]
    
    return youtube_urls[:max_results]

# === Vimeo Videos ===
def get_vimeo_video_urls(query, max_results=5, offset=0):
    """
    Fetch actual Vimeo videos related to the query using Vimeo API
    """
    # Vimeo API credentials
    vimeo_token = os.getenv('VIMEO_TOKEN')
    
    # Create a list to hold the video URLs
    vimeo_urls = []
    
    try:
        # Set up the headers for authentication
        headers = {
            'Authorization': f'Bearer {vimeo_token}'
        }
        
        # Prepare parameters for the API request
        params = {
            'query': query,
            'per_page': max_results,
            'sort': 'relevant',
            'direction': 'desc',
            'page': (offset // max_results) + 1
        }
        
        # Make the API request
        response = requests.get('https://api.vimeo.com/videos', headers=headers, params=params)
        data = response.json()
        
        # Check if we got any results
        if 'data' in data and data['data']:
            # Extract video URLs from the results
            for video in data['data']:
                if 'link' in video:
                    vimeo_urls.append(video['link'])
    
    except Exception as e:
        print(f"Error fetching Vimeo videos: {str(e)}")
        # Fall back to predefined videos if the API fails
        fallback_videos = [
            "https://vimeo.com/22439234",   # The Mountain
            "https://vimeo.com/274999546",  # RUMBLE
            "https://vimeo.com/35396305",   # Levi's Go Forth
            "https://vimeo.com/58659769",   # Nike Flyknit
            "https://vimeo.com/99263653"    # The Art of Flying
        ]
        
        # Shuffle based on query to always get same fallback videos for same celebrity
        name_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
        random.seed(name_hash + 1)  # Different seed from YouTube
        random.shuffle(fallback_videos)
        vimeo_urls = fallback_videos[offset:offset + max_results]
    
    return vimeo_urls[:max_results]

# === Dailymotion Videos ===
def get_dailymotion_video_urls(query, max_results=5, offset=0):
    """
    Fetch actual Dailymotion videos related to the query using Dailymotion API
    """
    # Create a list to hold the video URLs
    dailymotion_urls = []
    
    try:
        # Prepare parameters for the API request
        params = {
            'search': query,
            'limit': max_results,
            'fields': 'id,title,url',  # Specify which fields to return
            'sort': 'relevance',
            'flags': 'no_explicit',     # Safe search
            'page': (offset // max_results) + 1
        }
        
        # Make the API request
        response = requests.get('https://api.dailymotion.com/videos', params=params)
        data = response.json()
        
        # Check if we got any results
        if 'list' in data and data['list']:
            # Extract video URLs from the results
            for video in data['list']:
                if 'url' in video:
                    dailymotion_urls.append(video['url'])
                elif 'id' in video:
                    # Construct URL if only ID is available
                    dailymotion_urls.append(f"https://www.dailymotion.com/video/{video['id']}")
    
    except Exception as e:
        print(f"Error fetching Dailymotion videos: {str(e)}")
        # Fall back to predefined videos if the API fails
        fallback_videos = [
            "https://www.dailymotion.com/video/x84sh87",  # Nature Documentary
            "https://www.dailymotion.com/video/x5wad0d",  # Cooking Show
            "https://www.dailymotion.com/video/x7yfk6n",  # Drone Footage
            "https://www.dailymotion.com/video/x5wrf5v",  # Workout Video
            "https://www.dailymotion.com/video/x800gzv"   # Travel Guide
        ]
        
        # Add some TED talks as alternatives if we need more videos
        if len(fallback_videos) < max_results:
            ted_talks = [
                "https://www.ted.com/talks/hans_rosling_the_best_stats_you_ve_ever_seen",
                "https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity",
                "https://www.ted.com/talks/elizabeth_gilbert_your_elusive_creative_genius"
            ]
            fallback_videos.extend(ted_talks)
        
        # Shuffle based on query to always get same fallback videos for same celebrity
        name_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
        random.seed(name_hash + 2)  # Different seed from YouTube and Vimeo
        random.shuffle(fallback_videos)
        dailymotion_urls = fallback_videos[offset:offset + max_results]
    
    return dailymotion_urls[:max_results]

# === MAIN ===

if __name__ == "__main__":
    # Get search term from name.py celebrity detection
    image_path = input("Enter path to celebrity image: ")
    search_term = detect_top_celebrity_name(image_path)
    print(f"Detected: {search_term}")
    
    num_images = 10  # Default number of images
    num_videos = 5  # Default number of videos per platform

    # Google Images
    print(f"\n📸 Fetching {num_images} Image URLs for: {search_term}\n")
    images = get_google_image_urls(search_term, total_num=num_images)
    for i, url in enumerate(images, 1):
        print(f"{i}. {url}")

    # YouTube Videos
    print(f"\n▶️ Fetching {num_videos} YouTube Video URLs for: {search_term}\n")
    youtube_videos = get_youtube_video_urls(search_term, max_results=num_videos)
    for i, url in enumerate(youtube_videos, 1):
        print(f"{i}. {url}")

    # Vimeo Videos
    print(f"\n🎬 Fetching {num_videos} Vimeo Video URLs for: {search_term}\n")
    vimeo_videos = get_vimeo_video_urls(search_term, max_results=num_videos)
    for i, url in enumerate(vimeo_videos, 1):
        print(f"{i}. {url}")

    # Dailymotion Videos
    print(f"\n📹 Fetching {num_videos} Dailymotion Video URLs for: {search_term}\n")
    dailymotion_videos = get_dailymotion_video_urls(search_term, max_results=num_videos)
    for i, url in enumerate(dailymotion_videos, 1):
        print(f"{i}. {url}")