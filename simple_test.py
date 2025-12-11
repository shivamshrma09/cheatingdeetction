import requests

# Test with sample code only (no files needed)
def test_code_analysis():
    url = "https://shivaaaahdjdnd-anticheating.hf.space/post-interview-analysis"
    
    # Sample suspicious code
    suspicious_code = """
def solution(nums):
    # This function efficiently solves the two sum problem
    # using a hash map approach for optimal time complexity
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    return []
"""
    
    data = {'code_text': suspicious_code}
    
    try:
        response = requests.post(url, data=data)
        print("Response Status:", response.status_code)
        print("Analysis Result:", response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_code_analysis()