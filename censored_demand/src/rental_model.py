from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan


class RentalInventory:
	"""A model of rental inventory, modeling stock levels as returns and rentals occur each day.
	Currently supports a single product
	"""
	def __init__(self, n_products: int = 1, policies: Dict = {}):
		self.n_products = n_products
		self.policies = policies
		# Rentals that are out with customers are stored as an array, where the index corresponds with time, 
		# and the value corresponds with the number of rentals from that time that are still out with customers
		# max_periods is the total number of periods to log
		self.max_periods = 10000

	def model(self, init_state: Dict, start_time: int, end_time: int):
		_, ys = scan(
			self.model_single_day, 
			init=init_state, 
			xs=jnp.arange(start_time, end_time)
		)
		return ys

	def model_single_day(self, state, time):
		"""
		"""
		state_next = dict()

		# Simulate Returns
		returns = self.returns_model(state['existing_rentals'], time)
		state_next['starting_stock'] = numpyro.deterministic("starting_stock", state['ending_stock']+returns.sum() + self.apply_policy(time))

		# Simulate Rentals, incorporate them into the next state
		rentals = self.demand_model(available_stock=state_next['starting_stock'], time=time)
		state_next['ending_stock'] = numpyro.deterministic("ending_stock", state_next['starting_stock'] - rentals.sum())
		state_next['existing_rentals'] = numpyro.deterministic("existing_rentals", state['existing_rentals'] - returns + rentals)
		return state_next, rentals

	def returns_model(self, existing_rentals, time):
		theta = numpyro.sample("theta", dist.Normal(2.9, 0.01))
		sigma = numpyro.sample("sigma", dist.TruncatedNormal(0.7, 0.01, low=0))
		return_dist = dist.LogNormal(theta, sigma)

		# For each day of historical rentals that are currently rented, calculate how long they've been rented for
		rental_durations = (time-jnp.arange(self.max_periods))
		discrete_hazards = jnp.where(
			# If rental duration is nonnegative,
			rental_durations>0,
			# Use those rental durations to calculate a return rate, using a discrete interval hazard function
			RentalInventory.hazard_func(jnp.clip(rental_durations, a_min=0), dist=return_dist ),
			# Otherwise, return rate is 0
			0
		)
		# returns_sampled = numpyro.sample("returns_sampled", dist.Poisson(discrete_hazards*existing_rentals))
		# returns = numpyro.deterministic("returns", jnp.clip(returns_sampled, 0, existing_rentals.astype("int32")))
		returns = numpyro.sample("returns", dist.Binomial(existing_rentals.astype("int32"),probs=discrete_hazards))
		total_returns = numpyro.deterministic("total_returns", returns.sum())
		return returns

	def demand_model(self, available_stock, time):
		lambd = numpyro.sample("lambd", dist.Normal(10, 0.01))
		unconstrained_rentals = numpyro.sample("unconstrained_rentals", dist.Poisson(lambd))
		rentals = numpyro.deterministic("rentals", jnp.clip(unconstrained_rentals, a_min=0, a_max=available_stock ))
		rentals_as_arr = ( time == jnp.arange(self.max_periods) )*rentals
		return rentals_as_arr

	@staticmethod
	def hazard_func(t, dist):
		"""Discrete interval hazard function - aka the probability of a return occurring on a single date
		"""
		return (dist.cdf(t+1)-dist.cdf(t))/(1-dist.cdf(t))

	def apply_policy(self, time):
		return self.policies[time]

