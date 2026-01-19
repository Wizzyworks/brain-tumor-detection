import requests
import os

def test_api():
    url = "http://localhost:8001/predict"
    # Using 'images/tumor/test3.jpg' as established in infer.py
    image_path = "images/tumor/test3.jpg" 
    
    if not os.path.exists(image_path):
        print(f"Error: Test image {image_path} not found.")
        return

    print(f"Testing API at {url} with {image_path}...")
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("test3.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            print("✅ API Success!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ API Failed with status {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")

if __name__ == "__main__":
    test_api()
