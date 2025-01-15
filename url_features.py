import re
from urllib.parse import urlparse
import tldextract
import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle

class URLFeatureExtractor:
    def __init__(self):
        """Initialize the URL feature extractor"""
        # Define the expected feature names in the correct order
        self.feature_names = [
            'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex',
            'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength',
            'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
            'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
            'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
            'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
            'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
            'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
            'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
            'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
            'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
            'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef',
            'NoOfEmptyRef', 'NoOfExternalRef'
        ]
    
    def extract_features(self, url):
        """Extract features from a given URL"""
        features = {}
        
        # Initialize all features with default values
        for feature in self.feature_names:
            features[feature] = 0
        
        # Basic URL features
        features['URLLength'] = len(url)
        
        # Parse URL
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        # Domain features
        domain = extracted.domain + '.' + extracted.suffix
        features['DomainLength'] = len(domain)
        features['IsDomainIP'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # TLD features
        features['TLDLength'] = len(extracted.suffix)
        features['NoOfSubDomain'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
        
        # Character analysis
        features['NoOfLettersInURL'] = len(re.findall(r'[a-zA-Z]', url))
        features['LetterRatioInURL'] = features['NoOfLettersInURL'] / features['URLLength']
        features['NoOfDegitsInURL'] = len(re.findall(r'\d', url))
        features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / features['URLLength']
        
        # Special characters
        features['NoOfEqualsInURL'] = url.count('=')
        features['NoOfQMarkInURL'] = url.count('?')
        features['NoOfAmpersandInURL'] = url.count('&')
        special_chars = re.findall(r'[^a-zA-Z0-9=?&]', url)
        features['NoOfOtherSpecialCharsInURL'] = len(special_chars)
        features['SpacialCharRatioInURL'] = len(special_chars) / features['URLLength']
        
        # Security features
        features['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0
        
        try:
            # Make a request to the URL with a timeout
            response = requests.get(url, timeout=5, verify=False, allow_redirects=True)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # HTML features
            features['HasTitle'] = 1 if soup.title else 0
            features['HasFavicon'] = 1 if soup.find('link', rel='icon') else 0
            features['HasDescription'] = 1 if soup.find('meta', {'name': 'description'}) else 0
            features['NoOfImage'] = len(soup.find_all('img'))
            features['NoOfPopup'] = len(soup.find_all(['popup', 'dialog']))
            features['NoOfiFrame'] = len(soup.find_all('iframe'))
            features['HasPasswordField'] = 1 if soup.find('input', {'type': 'password'}) else 0
            features['HasHiddenFields'] = 1 if soup.find('input', {'type': 'hidden'}) else 0
            features['HasSubmitButton'] = 1 if soup.find(['button', 'input'], {'type': 'submit'}) else 0
            
            # Count various elements
            features['NoOfCSS'] = len(soup.find_all('link', {'rel': 'stylesheet'}))
            features['NoOfJS'] = len(soup.find_all('script'))
            
            # Links analysis
            all_links = soup.find_all('a')
            features['NoOfSelfRef'] = len([link for link in all_links if link.get('href', '').startswith('#')])
            features['NoOfEmptyRef'] = len([link for link in all_links if not link.get('href')])
            features['NoOfExternalRef'] = len([link for link in all_links if link.get('href', '').startswith('http')])
            
            # Check for sensitive words
            page_text = soup.get_text().lower()
            features['Bank'] = 1 if any(word in page_text for word in ['bank', 'banking', 'account']) else 0
            features['Pay'] = 1 if any(word in page_text for word in ['pay', 'payment', 'credit']) else 0
            features['Crypto'] = 1 if any(word in page_text for word in ['crypto', 'bitcoin', 'wallet']) else 0
            
            # Additional features
            features['HasCopyrightInfo'] = 1 if re.search(r'©|copyright', page_text) else 0
            features['HasSocialNet'] = 1 if any(social in str(soup) for social in ['facebook', 'twitter', 'instagram']) else 0
            
            # Line analysis
            lines = response.text.splitlines()
            features['LineOfCode'] = len(lines)
            features['LargestLineLength'] = max(len(line) for line in lines) if lines else 0
            
        except Exception as e:
            # Request failed - features remain as default values
            pass
            
        return features

    def transform_features(self, features_dict):
        """Transform features dictionary into model input format"""
        # Load the scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Convert dictionary to array in the correct order
        feature_array = np.array([features_dict[feature] for feature in self.feature_names]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        return scaled_features
