import asyncio
import aiohttp
import logging
import time
from web3 import Web3
from dotenv import load_dotenv
import os
from decimal import Decimal
from aiolimiter import AsyncLimiter
import random
from cachetools import TTLCache
from cachetools import TTLCache

# Define the cache with a Time-To-Live (TTL) of 60 seconds
gas_price_cache = TTLCache(maxsize=1, ttl=60)  # Cache gas price for 60 seconds

# Add this line at the global scope to initialize the nonce lock
nonce_lock = asyncio.Lock()

# ======== Logging Configuration ========
# Set up logging to both console and a log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_output.log"),  # Save logs to a file
        logging.StreamHandler()  # Also output to console
    ]
)

# ======== Environment Setup ========
load_dotenv()
ETHEREUM_RPC_URL = os.getenv('ETHEREUM_RPC_URL')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
ONEINCH_API_KEY = os.getenv('ONEINCH_API_KEY')
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv('YOUR_WALLET_ADDRESS'))

# ======== Web3 Initialization ========
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider

w3 = AsyncWeb3(AsyncHTTPProvider(ETHEREUM_RPC_URL))

# ======== API Endpoints ========
BASE_URL = 'https://api.1inch.dev/swap/v6.0/1'
SWAP_URL = f'{BASE_URL}/swap'
GAS_PRICE_URL = 'https://api.1inch.dev/gas-price/v1.4/1'

# ======== Expanded Token List ========
TOKENS = {
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # Wrapped Ether
    'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',   # DAI Stablecoin
    'USDC': '0xA0b86991C6218b36c1d19D4A2e9Eb0CE3606EB48',  # USD Coin
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # Tether USD
    'TUSD': '0x0000000000085d4780B73119b644AE5ecd22b376',  # TrueUSD
    'PAXG': '0x45804880De22913dAFE09f4980848ECE6EcbAf78',  # PAX Gold
    'COMP': '0xc00e94Cb662C3520282E6f5717214004A7f26888',  # Compound
    'AAVE': '0x7Fc66500c84A76AD7e9c93437bFc5Ac33E2DDaE9',  # AAVE Token
    'SUSHI': '0x6B3595068778DD592e39A122f4F5a5cf09C90fE2', # SushiSwap
    'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',  # Chainlink
    'MKR': '0x9f8F72aA9304c8B593d555F12ef6589cC3A579A2',   # Maker
    'YFI': '0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e',   # Yearn Finance
    'SNX': '0xC011A72400E58ecD99Ee497CF89E3775d4bd732F',   # Synthetix
    'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',   # Uniswap
    'CRV': '0xD533a949740bb3306d119CC777fa900bA034cd52',   # Curve DAO Token
    'SXP': '0x8CE9137d6220fce37aB2b9E571A6734F6c1dF6f5',   # Swipe
    'BAT': '0x0D8775F648430679A709E98d2b0Cb6250d2887EF',   # Basic Attention Token
    '1INCH': '0x111111111117dC0aa78b770fA6A738034120C302'  # 1inch
}

# ======== API Configuration ========
headers = {
    'Authorization': f'Bearer {ONEINCH_API_KEY}',
    'accept': 'application/json',
}

# Convert gas price to decimal for compatibility
#gas_price_decimal = Decimal(gas_price)
#gas_cost = gas_price_decimal * Decimal(estimated_gas_usage)

# Add this with your other constants
ESTIMATED_GAS_USAGE = 100000  # Estimated gas for a typical transaction
# ======== Constants ========
SLIPPAGE = 1  # Default slippage tolerance percentage
MIN_ETH_RESERVE = Web3.to_wei(0.01, 'ether')  # Minimum ETH to keep for gas
MAX_GAS_PRICE = Web3.to_wei(100, 'gwei')      # Maximum acceptable gas price
estimated_gas_usage = 100000  # Estimated gas for a typical transaction

# Exponential backoff for rate limit handling
def exponential_backoff(max_retries=5, base_delay=2):  # Increase base delay
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientError as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Request failed. Retrying in {delay:.2f} seconds. Error: {e}")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

def convert_amount_to_minimal_units(amount, decimals):
    # Ensure amount is properly scaled based on the token's decimals
    return int(amount * (10 ** decimals))

# Example of calling the function for USDC (6 decimals)
usdc_amount_in_wei = convert_amount_to_minimal_units(0.150823763, 6)  # Amount in USDC minimal units


def get_token_decimals(token_address):
    decimals_map = {
        '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE': 18,  # ETH
        '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 18,  # WETH
        '0x6B175474E89094C44Da98b954EedeAC495271d0F': 18,  # DAI
        '0xA0b86991C6218b36c1d19D4A2e9Eb0CE3606EB48': 6,   # USDC
        '0xdAC17F958D2ee523a2206206994597C13D831ec7': 6,   # USDT
        '0x0000000000085d4780B73119b644AE5ecd22b376': 18,  # TUSD
        '0x45804880De22913dAFE09f4980848ECE6EcbAf78': 18,  # PAXG
        '0xc00e94Cb662C3520282E6f5717214004A7f26888': 18,  # COMP
        '0x4E15361FD6b4BB609Fa63C81A2be19d873717870': 18,  # FTM
        '0x3472A5A71965499acd81997a54BBA8D852C6E53d': 18,  # BADGER
        '0x853d955aCEf822Db058eb8505911ED77F175b99e': 18,  # FRAX
        '0x4fabb145d64652a948d72533023f6e7a623c7c53': 18,  # BUSD
        '0x57ab1ec28d129707052df4df418d58a2d46d5f51': 18,  # sUSD
        '0x056fd409e1d7a124bd7017459dfea2f387b6d5cd': 2,   # GUSD
        '0x68749665ff8d2d112fa859aa293f07a622782f38': 6,   # XAUT
        '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9': 18,  # AAVE
        '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984': 18,  # UNI
        '0xD533a949740bb3306d119CC777fa900bA034cd52': 18,  # CRV
        '0x6b3595068778dd592e39a122f4f5a5cf09c90fe2': 18,  # SUSHI
        '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0': 18,  # MATIC
    }
    return decimals_map.get(token_address, 18)  # Default to 18 if not found


# Fetch Gas Price from 1inch API with dynamic adjustment
@exponential_backoff()
async def get_gas_price(session):
    cache_key = 'gas_price'
    
    # Check gas price cache
    if cache_key in gas_price_cache:
        return gas_price_cache[cache_key]

    async with rate_limit:
        try:
            async with session.get(GAS_PRICE_URL, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    gas_price = int(data['medium']['maxFeePerGas'])
                    
                    if gas_price < Web3.to_wei(1, 'gwei') or gas_price > Web3.to_wei(300, 'gwei'):
                        logging.warning(f"Gas price unusually high or low: {gas_price}. Adjusting to a dynamic range.")
                        gas_price = max(Web3.to_wei(2, 'gwei'), min(gas_price, Web3.to_wei(200, 'gwei')))
                    gas_price_cache[cache_key] = gas_price  # Cache gas price
                    return gas_price
                else:
                    logging.error(f"Failed to fetch gas price. Status code: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Unexpected error fetching gas price: {e}")
            return None

# Can execute trade
async def can_execute_trade(amount_to_trade, gas_price, estimated_gas_usage, num_of_swaps):
    """
    Ensures there is enough WETH to cover the trade and all gas costs for multiple swaps.
    """
    total_gas_cost = gas_price * estimated_gas_usage * num_of_swaps
    if amount_to_trade > total_gas_cost + MIN_ETH_RESERVE:
        return True
    else:
        logging.warning(f"Not enough WETH for the trade. Required gas cost: {total_gas_cost} wei.")
        return False


# Fetch ETH Balance (for gas fee purposes)
async def get_eth_balance():
    try:
        eth_balance = await w3.eth.get_balance(WALLET_ADDRESS)
        eth_balance_ether = Web3.from_wei(eth_balance, 'ether')
        logging.info(f"ETH Balance: {eth_balance_ether} ETH")  # Log ETH balance
        return eth_balance
    except Exception as e:
        logging.error(f"Error fetching ETH balance: {e}")
        return None


# Slippage
def get_slippage(token_symbol):
    high_volatility_tokens = ['1INCH', 'AAVE', 'SUSHI', 'YFI']
    if token_symbol in high_volatility_tokens:
        return 2  # High volatility token, increase slippage
    return 1  # Default slippage for most tokens

# Get Nonce with logging and caching improvements
async def get_nonce():
    global nonce_cache
    try:
        async with nonce_lock:
            current_nonce = await w3.eth.get_transaction_count(WALLET_ADDRESS)
            if nonce_cache is None or current_nonce > nonce_cache:
                nonce_cache = current_nonce
                logging.info(f"Fetched new nonce: {nonce_cache}")
            else:
                logging.info(f"Using cached nonce: {nonce_cache}")
            
            return_nonce = nonce_cache  # Store current nonce for use
            nonce_cache += 1  # Increment only after it's assigned
            return return_nonce
    except Exception as e:
        logging.error(f"Error fetching nonce: {e}")
        return None

# Fetch ETH Balance (for gas fee purposes)
async def get_eth_balance():
    try:
        eth_balance = await w3.eth.get_balance(WALLET_ADDRESS)
        eth_balance_ether = Web3.from_wei(eth_balance, 'ether')
        logging.info(f"ETH Balance: {eth_balance_ether} ETH")
        return eth_balance
    except Exception as e:
        logging.error(f"Error fetching ETH balance: {e}")
        return None


# Fetch Wallet Balance (WETH Only)
async def get_weth_balance():
    try:
        weth_contract = w3.eth.contract(address=TOKENS['WETH'], abi=[{
            'constant': True,
            'inputs': [{'name': '_owner', 'type': 'address'}],
            'name': 'balanceOf',
            'outputs': [{'name': 'balance', 'type': 'uint256'}],
            'type': 'function',
        }])
        balance = await weth_contract.functions.balanceOf(WALLET_ADDRESS).call()
        weth_balance = Web3.from_wei(balance, 'ether')
        if weth_balance < 0.01:
            logging.warning(f"WETH balance is very low: {weth_balance} WETH")
        else:
            logging.info(f"WETH Balance: {weth_balance} WETH")
        return weth_balance
    except Exception as e:
        logging.error(f"Error fetching WETH balance: {e}")
        return None

# Approve Token for 1inch Swap with TTL Cache
async def approve_token(token_address):
    global allowance_cache, allowance_cache_time

    ttl = 3600

    try:
        if token_address in allowance_cache and (time.time() - allowance_cache_time.get(token_address, 0)) < ttl:
            allowance = allowance_cache[token_address]
            logging.info(f"Using cached allowance for {token_address}: {allowance}")
        else:
            erc20_abi = [
                {"constant": False, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
                {"constant": True, "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "remaining", "type": "uint256"}], "type": "function"}
            ]
            contract = w3.eth.contract(address=token_address, abi=erc20_abi)
            allowance = await contract.functions.allowance(WALLET_ADDRESS, '0x11111112542d85b3ef69ae05771c2dccff4faa26').call()
            
            if allowance is None or allowance == 0:
                logging.error(f"Allowance fetch failed for {token_address}. Unexpected allowance value: {allowance}")
                return None
            
            allowance_cache[token_address] = allowance
            allowance_cache_time[token_address] = time.time()
            logging.info(f"Fetched new allowance for {token_address}: {allowance}")

        if allowance < Web3.to_wei(1000, 'ether'):
            logging.info(f"Approving {token_address} for 1inch swap...")
            approve_tx = await contract.functions.approve(
                '0x11111112542d85b3ef69ae05771c2dccff4faa26',
                Web3.to_wei(1000000, 'ether')
            ).buildTransaction({
                'from': WALLET_ADDRESS,
                'nonce': await get_nonce(),
                'gas': 100000,
                'gasPrice': await w3.eth.gas_price
            })

            signed_approve_tx = w3.eth.account.sign_transaction(approve_tx, PRIVATE_KEY)
            tx_hash = await w3.eth.send_raw_transaction(signed_approve_tx.raw_transaction)
            logging.info(f"Approval transaction sent: {tx_hash.hex()}")

            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Approval transaction receipt: {receipt}")

            allowance_cache[token_address] = Web3.to_wei(1000000, 'ether')
    except Exception as e:
        logging.error(f"Error approving token {token_address}: {e}")

# Get Swap DATA with retry, logging, and detailed transaction info
# Rate limit monitoring variables
rate_limit_hit_count = 0
RATE_LIMIT_WARNING_THRESHOLD = 5  # Number of consecutive rate limit hits before slowing down
ADAPTIVE_BACKOFF_MULTIPLIER = 1.5  # Multiplier for increasing wait time after each rate limit hit
initial_backoff = 2  # Start with 2 seconds
swap_cache = TTLCache(maxsize=100, ttl=60)  # Cache swap data for 60 seconds

# Set the rate limit to 1 request per second
rate_limit = AsyncLimiter(1, 1)  # 1 request per second

# Use the limiter in your functions that call the API
# Use the limiter in your functions that call the API
@exponential_backoff()
async def get_swap_data(session, from_token, to_token, amount, slippage):
    cache_key = f"{from_token}_{to_token}_{amount}_{slippage}"
    
    # Check if the result is in the cache
    if cache_key in swap_cache:
        return swap_cache[cache_key]

    await asyncio.sleep(1)  # Ensure at least 1 second delay before each request
    async with rate_limit:
        params = {
            'fromTokenAddress': from_token,
            'toTokenAddress': to_token,
            'amount': str(int(amount)),  # Make sure amount is a string in minimal units
            'fromAddress': WALLET_ADDRESS,
            'slippage': slippage,
            'disableEstimate': 'false',
            'allowPartialFill': 'false',
        }

        logging.info(f"Requesting swap data with params: {params}")  # Log the parameters being sent

        async with session.get(SWAP_URL, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                result = data.get('tx')
                # Cache the result
                swap_cache[cache_key] = result
                return result
            else:
                logging.error(f"Error fetching swap data. Status code: {response.status}. Response: {await response.text()}")
                return None




# Ensure the swap amount is valid before making the request
def is_valid_amount(amount_in_wei):
    return amount_in_wei > 0

# Calculate potential profit for a given strategy with enhanced gas and error handling
MIN_PROFIT_MARGIN = Web3.to_wei(0.01, 'ether')  # Adjust as needed

async def calculate_profit(session, from_token, intermediate_token, to_token, amount):
    try:
        gas_price = await get_gas_price(session)
        if gas_price is None:
            logging.error("Failed to retrieve gas price, skipping profit calculation.")
            return 0

        slippage = get_slippage(intermediate_token)

        # First swap: from_token -> intermediate_token
        first_swap = await get_swap_data(session, from_token, intermediate_token, amount, slippage)
        if not first_swap:
            logging.error("First swap failed")
            return 0

        first_received = int(first_swap['value'])
        gas_used_first = int(first_swap['gas'])

        # Ensure the first received amount is valid
        if first_received <= 0:
            logging.error("Invalid received amount from the first swap, skipping the second swap.")
            return 0

        # Second swap: intermediate_token -> to_token
        second_swap = await get_swap_data(session, intermediate_token, to_token, first_received, slippage)
        if not second_swap:
            logging.error("Second swap failed")
            return 0

        second_received = int(second_swap['value'])
        gas_used_second = int(second_swap['gas'])

        third_swap = await get_swap_data(session, to_token, TOKENS['WETH'], second_received, slippage)
        if not third_swap:
            logging.error("Third swap failed")
            return 0

        final_received = int(third_swap['value'])
        gas_used_third = int(third_swap['gas'])

        total_gas_used = gas_used_first + gas_used_second + gas_used_third
        total_gas_cost = total_gas_used * gas_price

        profit = final_received - amount - total_gas_cost

        if profit < MIN_PROFIT_MARGIN:
            logging.warning(f"Profit {profit} is below the minimum margin {MIN_PROFIT_MARGIN}, skipping trade.")
            return 0

        logging.info(f"Calculated profit: {profit} for trade {from_token} -> {intermediate_token} -> {to_token}")
        return profit
    except Exception as e:
        logging.error(f"Error calculating profit: {e}")
        return 0


# Execute Swap with transaction retry and better error handling
async def execute_swap(session, from_token, to_token, amount):
    from_token = Web3.to_checksum_address(from_token)
    to_token = Web3.to_checksum_address(to_token)

    if from_token != TOKENS['WETH']:
        await approve_token(from_token)

    tx_data = await get_swap_data(session, from_token, to_token, amount, get_slippage(to_token))

    if not tx_data:
        logging.warning(f"Could not get swap data for {from_token} to {to_token}")
        return None

    nonce = await get_nonce()

    try:
        tx = {
            'to': Web3.to_checksum_address(tx_data['to']),
            'value': int(tx_data['value']),
            'gas': int(tx_data['gas']),
            'gasPrice': int(tx_data['gasPrice']),
            'nonce': nonce,
            'data': tx_data['data']
        }

        logging.info(f"Signing transaction: {from_token} -> {to_token}, amount: {amount}, nonce: {nonce}")

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logging.info(f"Transaction sent: {tx_hash.hex()}")

        # Wait for the transaction to be confirmed
        receipt = await wait_for_confirmation(tx_hash)
        if receipt['status'] == 1:
            logging.info(f"Transaction {tx_hash.hex()} confirmed with receipt: {receipt}")
        else:
            logging.error(f"Transaction {tx_hash.hex()} failed with receipt: {receipt}")
            return None

        return receipt
    except ValueError as e:
        logging.error(f"Transaction signing error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error executing swap: {e}")
        return None

# Wait for transaction confirmation
async def wait_for_confirmation(tx_hash):
    logging.info(f"Waiting for transaction {tx_hash.hex()} to be confirmed...")
    while True:
        try:
            receipt = await w3.eth.get_transaction_receipt(tx_hash)
            if receipt is not None:
                return receipt
            await asyncio.sleep(2)
        except Exception as e:
            logging.warning(f"Transaction {tx_hash.hex()} not yet confirmed. Retrying in 2 seconds...")
            await asyncio.sleep(2)

# ======== Arbitrage Strategies ========

# Simple Arbitrage (WETH to Another Token and Back)
async def execute_simple_arbitrage(session, from_token, to_token, amount):
    gas_price = await get_gas_price(session)
    if await can_execute_trade(amount, gas_price, estimated_gas_usage=100000, num_of_swaps=2):
        first_leg = await execute_swap(session, from_token, to_token, amount)
        if not first_leg:
            logging.warning(f"Simple arbitrage failed for {from_token} -> {to_token}")
            return

        second_leg = await execute_swap(session, to_token, TOKENS['WETH'], amount)
        if not second_leg:
            logging.warning(f"Simple arbitrage failed for {to_token} -> WETH")
            return

        logging.info(f"Simple arbitrage completed: {from_token} -> {to_token} -> WETH")
    else:
        logging.warning(f"Not enough WETH to execute simple arbitrage {from_token} -> {to_token}")

# Multi-Leg Arbitrage (WETH -> Intermediate -> Another Token -> WETH)
async def execute_multi_leg_arbitrage(session, from_token, intermediate_token, to_token, amount):
    gas_price = await get_gas_price(session)
    if await can_execute_trade(amount, gas_price, estimated_gas_usage=100000, num_of_swaps=3):
        first_leg = await execute_swap(session, from_token, intermediate_token, amount)
        if not first_leg:
            logging.warning(f"First leg failed for {from_token} -> {intermediate_token}")
            return

        second_leg = await execute_swap(session, intermediate_token, to_token, amount)
        if not second_leg:
            logging.warning(f"Second leg failed for {intermediate_token} -> {to_token}")
            return

        third_leg = await execute_swap(session, to_token, TOKENS['WETH'], amount)
        if not third_leg:
            logging.warning(f"Third leg failed for {to_token} -> WETH")
            return

        logging.info(f"Multi-leg arbitrage completed: {from_token} -> {intermediate_token} -> {to_token} -> WETH")
    else:
        logging.warning(f"Not enough WETH to execute multi-leg arbitrage {from_token} -> {intermediate_token} -> {to_token}")

# Triangular Arbitrage (WETH -> Token -> Another Token -> WETH)
async def execute_triangular_arbitrage(session, from_token, intermediate_token, to_token, amount):
    gas_price = await get_gas_price(session)
    if await can_execute_trade(amount, gas_price, estimated_gas_usage=100000, num_of_swaps=3):
        first_leg = await execute_swap(session, from_token, intermediate_token, amount)
        if not first_leg:
            logging.warning(f"First leg failed for {from_token} -> {intermediate_token}")
            return

        second_leg = await execute_swap(session, intermediate_token, to_token, amount)
        if not second_leg:
            logging.warning(f"Second leg failed for {intermediate_token} -> {to_token}")
            return

        third_leg = await execute_swap(session, to_token, TOKENS['WETH'], amount)
        if not third_leg:
            logging.warning(f"Third leg failed for {to_token} -> WETH")
            return

        logging.info(f"Triangular arbitrage completed: {from_token} -> {intermediate_token} -> {to_token} -> WETH")
    else:
        logging.warning(f"Not enough WETH to execute triangular arbitrage {from_token} -> {intermediate_token} -> {to_token}")

# Main Execution Loop with concurrent profit calculation and strategy execution
from decimal import Decimal

async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Fetch gas price
                gas_price = await get_gas_price(session)
                if gas_price is None:
                    logging.error("Failed to retrieve gas price. Waiting before retrying.")
                    await asyncio.sleep(60)
                    continue

                # Fetch WETH and ETH balances for trading and gas fees
                weth_balance = await get_weth_balance()
                eth_balance = await get_eth_balance()

                if weth_balance is None or eth_balance is None:
                    logging.error("Failed to fetch balances. Waiting before retrying.")
                    await asyncio.sleep(60)
                    continue

                # Example of logging terminal output
                logging.info(f"Fetched WETH Balance: {weth_balance}, ETH Balance: {eth_balance}")

                # Convert WETH balance to wei for calculations
                weth_balance_wei = Web3.to_wei(Decimal(weth_balance), 'ether')

                # Calculate gas cost
                estimated_gas_usage = ESTIMATED_GAS_USAGE
                gas_cost = Decimal(gas_price) * Decimal(estimated_gas_usage)

                # Define how much WETH to trade (in wei)
                trading_fraction = Decimal('0.5')  # Adjust as needed
                amount_to_trade_wei = (weth_balance_wei * trading_fraction) - gas_cost

                if amount_to_trade_wei <= 0:
                    logging.warning(f"Not enough WETH to trade after gas cost. WETH: {weth_balance}, Gas Cost: {Web3.from_wei(gas_cost, 'ether')} ETH")
                    await asyncio.sleep(60)
                    continue

                # Convert amount_to_trade back to ether for logging
                amount_to_trade_eth = Web3.from_wei(amount_to_trade_wei, 'ether')
                logging.info(f"Amount to trade: {amount_to_trade_eth} WETH")

                if gas_price > MAX_GAS_PRICE:
                    logging.warning("Gas price too high, skipping this cycle.")
                    await asyncio.sleep(60)
                    continue

                # List to gather all profit calculation tasks
                profit_tasks = []

                # Limit concurrent tasks to avoid rate limiting
                semaphore = asyncio.Semaphore(1)

                async def calculate_profit_with_limit(*args):
                    async with semaphore:
                        return await calculate_profit(*args)

                # Iterate over all tokens for arbitrage opportunities
                for from_token_symbol, from_token_address in TOKENS.items():
                    for to_token_symbol, to_token_address in TOKENS.items():
                        # Skip if from_token and to_token are the same
                        if from_token_symbol == to_token_symbol:
                            continue

                        # Convert amount to minimal units
                        amount_in_minimal_units = convert_amount_to_minimal_units(amount_to_trade_wei, get_token_decimals(from_token_address))

                        logging.info(f"Attempting to swap {from_token_symbol} -> {to_token_symbol} with amount {amount_in_minimal_units}")

                        # Add multi-leg arbitrage task
                        profit_tasks.append(calculate_profit_with_limit(session, from_token_address, TOKENS['WETH'], to_token_address, amount_in_minimal_units))

                        # Add triangular arbitrage task
                        profit_tasks.append(calculate_profit_with_limit(session, TOKENS['WETH'], from_token_address, to_token_address, amount_in_minimal_units))

                # Run all profit calculations concurrently
                profit_results = await asyncio.gather(*profit_tasks)

                # Find the most profitable arbitrage strategy
                best_profit = None
                best_strategy = None
                best_token = None

                for i, profit in enumerate(profit_results):
                    if profit > 0:
                        if best_profit is None or profit > best_profit:
                            best_profit = profit
                            best_strategy = i % 2  # Strategy (0: multi-leg, 1: triangular)
                            best_token = list(TOKENS.values())[i // 2]  # Find corresponding token

                if best_profit is not None:
                    # Execute the most profitable strategy
                    if best_strategy == 0:
                        logging.info(f"Executing Multi-Leg Arbitrage for {best_token}")
                        await execute_multi_leg_arbitrage(session, TOKENS['WETH'], TOKENS['USDC'], best_token, amount_in_minimal_units)
                    elif best_strategy == 1:
                        logging.info(f"Executing Triangular Arbitrage for {best_token}")
                        await execute_triangular_arbitrage(session, TOKENS['WETH'], TOKENS['USDC'], best_token, amount_in_minimal_units)
                else:
                    logging.info("No profitable arbitrage opportunities found in this cycle.")

                # Cooldown between cycles
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                logging.info("Process was interrupted.")
                break
            except Exception as e:
                logging.error(f"Error encountered: {e}")
                await asyncio.sleep(60)  # Increased wait time after an error

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script terminated by user.")