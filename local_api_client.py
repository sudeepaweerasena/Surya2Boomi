#!/usr/bin/env python3
"""
Solar Flare Forecasting API Client

This script communicates with the Google Colab API to request 24-hour solar flare forecasts.

Usage:
    python local_api_client.py --check-health
    python local_api_client.py --forecast
    python local_api_client.py --forecast --output results.json
"""

import requests
import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# IMPORTANT: Update these values after running the Colab notebook
NGROK_URL = "https://preceptive-reflexively-stephen.ngrok-free.dev"
API_KEY = "CZ1r7dDVsE7Y-cn4FlXxVlRi3CHzv3WRTq2Qpy5RjXU"

# ============================================================================
# API CLIENT CLASS
# ============================================================================

class SolarFlareAPIClient:
    """Client for interacting with the Solar Flare Forecasting API"""
    
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def health_check(self):
        """Check if the API is running"""
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ API is healthy!")
                print(f"   Service: {data.get('service')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Device: {data.get('device')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {str(e)}")
            print(f"\n⚠️  Make sure:")
            print(f"   1. The Colab notebook is running")
            print(f"   2. The ngrok URL is correct: {self.base_url}")
            return False
    
    def get_status(self):
        """Get API status information"""
        try:
            url = f"{self.base_url}/status"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("📊 API Status:")
                print(f"   Model loaded: {data.get('model_loaded')}")
                print(f"   Device: {data.get('device')}")
                print(f"   Data available: {data.get('data_available')}")
                print(f"   Ready for inference: {data.get('ready_for_inference')}")
                return data
            elif response.status_code == 401:
                print("❌ Authentication failed. Check your API key.")
                return None
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {str(e)}")
            return None
    
    def get_forecast(self):
        """Request a 24-hour solar flare forecast"""
        try:
            print("🔮 Requesting 24-hour solar flare forecast...")
            print("   This may take a few seconds...\n")
            
            url = f"{self.base_url}/forecast"
            response = requests.post(url, headers=self.headers, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    print("✅ Forecast generated successfully!\n")
                    self._display_forecast(data)
                    return data
                else:
                    print(f"❌ Forecast failed: {data.get('message')}")
                    return None
                    
            elif response.status_code == 401:
                print("❌ Authentication failed. Check your API key.")
                return None
            else:
                print(f"❌ Forecast request failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('message', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Request timeout. The model inference is taking too long.")
            print("   Try again in a moment.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {str(e)}")
            return None
    
    def _display_forecast(self, forecast_data):
        """Display forecast results in a readable format"""
        
        print("=" * 80)
        print("🌞 SOLAR FLARE FORECAST - NEXT 24 HOURS")
        print("=" * 80)
        
        print(f"\n📅 Generated: {forecast_data.get('forecast_generated_at')}")
        print(f"🔭 Horizon: {forecast_data.get('forecast_horizon')}")
        
        model_info = forecast_data.get('model_info', {})
        print(f"🤖 Model: {model_info.get('name', 'Unknown')}")
        print(f"💻 Device: {model_info.get('device', 'Unknown')}")
        
        forecasts = forecast_data.get('forecasts', [])
        
        if forecasts:
            print(f"\n📊 Hourly Predictions ({len(forecasts)} hours):")
            print("-" * 80)
            print(f"{'Hour':>4} | {'Timestamp':^22} | {'Flare %':>7} | {'Class':>5} | {'Status':>15}")
            print("-" * 80)
            
            for fc in forecasts[:12]:  # Show first 12 hours in detail
                hour = fc.get('hour', 0)
                timestamp = fc.get('timestamp', '')[:19]  # Trim to readable format
                flare_prob = fc.get('flare_probability', 0) * 100
                flare_class = fc.get('flare_class', 'N/A')
                
                # Determine status emoji
                if flare_prob > 70:
                    status = "🔴 High Risk"
                elif flare_prob > 40:
                    status = "🟡 Moderate"
                else:
                    status = "🟢 Low Risk"
                
                print(f"{hour:4d} | {timestamp} | {flare_prob:6.2f}% | {flare_class:>5} | {status}")
            
            if len(forecasts) > 12:
                print(f"\n   ... and {len(forecasts) - 12} more hours")
            
            # Summary statistics
            avg_prob = sum(fc.get('flare_probability', 0) for fc in forecasts) / len(forecasts) * 100
            max_prob = max(fc.get('flare_probability', 0) for fc in forecasts) * 100
            max_hour = max(forecasts, key=lambda x: x.get('flare_probability', 0)).get('hour', 0)
            
            print("\n" + "=" * 80)
            print("📈 SUMMARY")
            print("=" * 80)
            print(f"   Average flare probability: {avg_prob:.2f}%")
            print(f"   Maximum flare probability: {max_prob:.2f}% (at hour {max_hour})")
            print(f"   Risk assessment: {'HIGH' if max_prob > 70 else 'MODERATE' if max_prob > 40 else 'LOW'}")
            print("=" * 80 + "\n")
    
    def save_forecast(self, forecast_data, output_path):
        """Save forecast data to a file"""
        try:
            output_path = Path(output_path)
            
            # Determine format based on extension
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(forecast_data, f, indent=2)
                print(f"💾 Forecast saved to: {output_path}")
                
            elif output_path.suffix.lower() == '.csv':
                import csv
                forecasts = forecast_data.get('forecasts', [])
                
                with open(output_path, 'w', newline='') as f:
                    if forecasts:
                        writer = csv.DictWriter(f, fieldnames=forecasts[0].keys())
                        writer.writeheader()
                        writer.writerows(forecasts)
                print(f"💾 Forecast saved to: {output_path}")
                
            else:
                # Default to JSON
                output_path = output_path.with_suffix('.json')
                with open(output_path, 'w') as f:
                    json.dump(forecast_data, f, indent=2)
                print(f"💾 Forecast saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving forecast: {str(e)}")
            return False

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Solar Flare Forecasting API Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check API health:
    %(prog)s --check-health
    
  Get forecast and display:
    %(prog)s --forecast
    
  Get forecast and save to file:
    %(prog)s --forecast --output forecast_results.json
    %(prog)s --forecast --output forecast_results.csv
    
  Get status:
    %(prog)s --status
    
Configuration:
  Edit the NGROK_URL and API_KEY variables at the top of this script
  to match the values from your running Colab notebook.
        """
    )
    
    parser.add_argument('--check-health', action='store_true',
                       help='Check if the API is running')
    parser.add_argument('--status', action='store_true',
                       help='Get API status information')
    parser.add_argument('--forecast', action='store_true',
                       help='Request a 24-hour solar flare forecast')
    parser.add_argument('--output', '-o', type=str,
                       help='Save forecast to file (JSON or CSV)')
    parser.add_argument('--url', type=str,
                       help='Override the default ngrok URL')
    parser.add_argument('--api-key', type=str,
                       help='Override the default API key')
    
    args = parser.parse_args()
    
    # Use provided URL/key or defaults
    url = args.url or NGROK_URL
    api_key = args.api_key or API_KEY
    
    # Validate configuration
    if url == "https://your-ngrok-url.ngrok-free.app":
        print("❌ Error: Please update the NGROK_URL in this script")
        print("   Or provide it with --url flag")
        print("\n   Example:")
        print(f"   python {sys.argv[0]} --forecast --url https://xxxx.ngrok-free.app")
        sys.exit(1)
    
    if api_key == "your-api-key-here":
        print("❌ Error: Please update the API_KEY in this script")
        print("   Or provide it with --api-key flag")
        sys.exit(1)
    
    # Create client
    client = SolarFlareAPIClient(url, api_key)
    
    # Execute requested action
    if args.check_health:
        client.health_check()
        
    elif args.status:
        client.get_status()
        
    elif args.forecast:
        forecast = client.get_forecast()
        
        if forecast and args.output:
            client.save_forecast(forecast, args.output)
    
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --check-health to verify the connection")

if __name__ == '__main__':
    main()
