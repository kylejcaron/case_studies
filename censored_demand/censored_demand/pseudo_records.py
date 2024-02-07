"""These are helper tools for making fake records from the simulation results. 
Ideally the simulation would generate these as the simulation runs, but its not performant, 
so fake records are made afterwards from the simulated results.

The code is pretty ugly, I'd recommend not reading into this too much. 
"""
from typing import Tuple
import sys
import argparse
import pandas as pd
import numpy as np
import uuid
import jax
import jax.numpy as jnp
import numpyro
from censored_demand.rental_model import PoissonDemandInventory, MultinomialDemandInventory
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(			
	"--demand_type", 
	"-d",
	default='multinomial',
	type=str,
	help="The model type. Can be poisson or multinomial"
)


def simulate_buy_amounts(utility, j_products, min_buy=20, max_buy=300):
    """This function simulates semi-informed buy amounts - 80% the time the buy is correlated to utility (but typically underbought)
    and the rest of the time the buy is randomly informed
    """
    X = np.random.gamma( utility+np.abs(utility.min()) + 5 ,0.2,size=j_products)
    semi_informed_depth=(min_buy + X/X.max()*max_buy) // 10*10

    is_informed_buy = np.random.binomial(1, 0.8, size=j_products)
    random_depth = np.random.choice(range(min_buy, max_buy+1,10), size=j_products)
    buy_amounts = np.where(is_informed_buy==True, semi_informed_depth, random_depth)
    return buy_amounts
    


def make_pseudo_rental_data(results, start_date='2022-04-01'):
	"""This takes a simulation results dictionary and converts it into pseudo realistic data
	"""
	j_products = results['rentals'].shape[-1]
	T = results['rentals'].shape[1]
	dates = pd.date_range(start_date, periods=T)

	all_stock_data = []
	all_rental_events = []
	for j in tqdm(range(j_products),total=j_products):
		daily_rentals = results['rentals'][...,j].ravel().astype(int)

		# Make a fake stock level dataset
		stock_data = pd.DataFrame({
			"product_id": f"product_{j}",
			"date": dates, 
			"starting_units": results['starting_stock'][...,j].ravel(),
			"ending_units": results['ending_stock'][...,j].ravel()
		})

		# Generate fake rental events
		rental_events = (
			pd.concat([
				generate_rental_events(daily_rentals[day_idx], date) 
				for day_idx, date in enumerate(dates)
			])
			.reset_index(drop=True)
			.assign(product_id=f"product_{j}")
		)

		# Generate fake return entries for the rental events
		for day_idx, date in enumerate(dates):
			
			# Pull the returns that occurred that day
			returns = np.array(results['returns'][0, day_idx, j, :T])

			# update the rental events to log the returns that occurred
			if returns.sum() > 0:
				rental_dates_of_returns = np.repeat(dates, returns)
				assert len(rental_dates_of_returns) == returns.sum()
				update_records_with_returns(rental_events, rental_dates_of_returns, date)
		
		all_stock_data.append(stock_data)
		all_rental_events.append(rental_events)

	return pd.concat(all_stock_data), pd.concat(all_rental_events)


def generate_rental_events(rentals, date) -> pd.DataFrame:
	"""Takes an count of rentals that occurred on a given date and generates
	fake entries for each one. 
	"""
	return (
		pd.DataFrame({
		# Generate an entry id for each rental that occurs
		"rental_id":[uuid.uuid4() for i in range(rentals)]
		})
		# assign the date of the rental
		.assign(date=date)
		# leave the date that the rental is returned as null
		.assign(return_date=pd.NaT)
	)


def update_records_with_returns(
	rental_events: pd.DataFrame, 
	rental_dates_of_returns: np.array, 
	date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""This takes a dataframe of rental events and updates the records if they 
	had returns on a given date
	"""

	for return_i in rental_dates_of_returns:

		# For each return that occurred, grab 1 corresponding entry 
		# based on when it was rented out and log its index
		idx = (
			rental_events
			.loc[lambda d: d.return_date.isnull()]
			.loc[lambda d: d.date == return_i]
			.head(1).loc[:, "return_date"]
		).index.values[0]

		# update that entry as having been returned on the inputted date
		rental_events.loc[idx, "return_date"] = date


if __name__ == '__main__':
	args = vars(parser.parse_args())

	T = 100
	j_products = 1000
	dates = pd.date_range('2022-04-01', periods=T, freq='D')

	print("Defining the inventory model....")
	# Simulate a reorder policy
	rng = np.random.default_rng(seed=99)
	reorder_amt = rng.choice(range(0,100+1,50), p=[0.9, 0.075,0.025], size=j_products)
	reorder_policy = (jnp.arange(T) == 60)*reorder_amt[:,None]

	# Declare the inventory model for simulation
	if args['demand_type'] == 'poisson':
		rental_inventory= PoissonDemandInventory(n_products=j_products, policies=reorder_policy)
	elif args['demand_type'] == 'multinomial':
		rental_inventory= MultinomialDemandInventory(n_products=j_products, policies=reorder_policy)
	else:
		raise ValueError("demand type must be one of [poisson, multinomial]")

	# Choose initial depths for each product
	initial_depth = simulate_buy_amounts(rental_inventory.U, j_products, min_buy = 25, max_buy=250)

	init_state = {
		"starting_stock": initial_depth,
		"ending_stock": initial_depth,
		# No pre-existing rentals for this product at the start of the simulation
		"existing_rentals": np.zeros((j_products,10000)),
	}
	
	print("Simulating Rental and Stock Activity....")
	# Simulate demand and stock for the product
	pred_fn = numpyro.infer.Predictive(
		rental_inventory.model, 
		num_samples = 1,
	)
	results = pred_fn(jax.random.PRNGKey(1), init_state, 0, T)

	print("Making pseudo-realistic records....")
	stock_data, rental_events = make_pseudo_rental_data(results, start_date = dates[0])

	print("Saving Results....")
	stock_data.to_csv("./data/stock_levels.csv", index=None)
	rental_events.to_csv("./data/rentals.csv", index=None)

	# Save purchase records
	pd.concat((
		
		# initial order volumes
		pd.DataFrame({
			"product_id":[f"product_{j}" for j in range(j_products)],
			"units":initial_depth,
			"date": dates[0],
			"order_type":"new_product"
		}),
		# reorder volumes
		pd.DataFrame({
			"product_id":[f"product_{j}" for j in range(j_products)],
			"units":reorder_amt,
			"date": dates[60],
			"order_type":"reorder"
		}).loc[lambda d: d.units>0],
	
	)).to_csv("./data/product_purchases.csv", index=None)

	# Same true parameters
	pd.DataFrame({
		"product_id":[f"product_{j}" for j in range(j_products)],
		"utility":rental_inventory.U
	}).to_csv("./data/true_params.csv", index=None)
	
	( 
		pd.DataFrame(results['unconstrained_demand'].squeeze(0), index=dates)
		.stack()
		.reset_index()
		.set_axis(['date', 'product_id', 'demand'],axis=1)
		.assign(product_id = lambda d: 'product_' + d.product_id.astype(str))
		.to_csv("./data/true_demand_daily.csv", index=None)
	)