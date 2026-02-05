"""
Fetch options data from Yahoo Finance and save as JSON for Rust SLV model.
"""
import yfinance as yf
import json
import sys
from datetime import datetime

def fetch_options(symbol="SPY", num_expiries=5):
    """Fetch options chain for a symbol."""
    ticker = yf.Ticker(symbol)

    # Get spot price
    try:
        spot = ticker.info.get('regularMarketPrice') or ticker.fast_info['lastPrice']
    except:
        spot = ticker.history(period='1d')['Close'].iloc[-1]

    print(f"{symbol} Spot: ${spot:.2f}")

    expirations = ticker.options[:num_expiries]
    print(f"Fetching {len(expirations)} expirations: {expirations}")

    surface_data = {
        "symbol": symbol,
        "spot": spot,
        "timestamp": datetime.now().isoformat(),
        "chains": []
    }

    for exp in expirations:
        print(f"  Fetching {exp}...")
        try:
            chain = ticker.option_chain(exp)

            calls_data = []
            puts_data = []

            # Filter for reasonable strikes (50% to 150% of spot)
            for _, row in chain.calls.iterrows():
                if 0.5 * spot <= row['strike'] <= 1.5 * spot:
                    calls_data.append({
                        "strike": row['strike'],
                        "bid": row['bid'] if row['bid'] > 0 else None,
                        "ask": row['ask'] if row['ask'] > 0 else None,
                        "last": row['lastPrice'] if row['lastPrice'] > 0 else None,
                        "iv": row['impliedVolatility'] if row['impliedVolatility'] > 0 else None,
                        "volume": int(row['volume']) if row['volume'] > 0 else 0,
                        "open_interest": int(row['openInterest']) if row['openInterest'] > 0 else 0
                    })

            for _, row in chain.puts.iterrows():
                if 0.5 * spot <= row['strike'] <= 1.5 * spot:
                    puts_data.append({
                        "strike": row['strike'],
                        "bid": row['bid'] if row['bid'] > 0 else None,
                        "ask": row['ask'] if row['ask'] > 0 else None,
                        "last": row['lastPrice'] if row['lastPrice'] > 0 else None,
                        "iv": row['impliedVolatility'] if row['impliedVolatility'] > 0 else None,
                        "volume": int(row['volume']) if row['volume'] > 0 else 0,
                        "open_interest": int(row['openInterest']) if row['openInterest'] > 0 else 0
                    })

            surface_data["chains"].append({
                "expiry": exp,
                "calls": calls_data,
                "puts": puts_data
            })

            print(f"    Calls: {len(calls_data)}, Puts: {len(puts_data)}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return surface_data


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    num_exp = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    data = fetch_options(symbol, num_exp)

    # Save to JSON
    output_file = f"D:/Stochastic local volatility/slv-options/data/{symbol.lower()}_options.json"

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {output_file}")

    # Print summary
    total_calls = sum(len(c['calls']) for c in data['chains'])
    total_puts = sum(len(c['puts']) for c in data['chains'])
    print(f"Total: {total_calls} calls, {total_puts} puts across {len(data['chains'])} expiries")


if __name__ == "__main__":
    main()
