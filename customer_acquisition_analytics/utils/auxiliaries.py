## Auxiliary functions

import pandas as pd
import numpy as np
from scipy.stats import lognorm
import math


import matplotlib.pyplot as plt
import plotly.express as px

from IPython.display import clear_output, display, Markdown, HTML

from utils.generate_dataset import session_end

def initial_clusters_deterministic(
        cohort_size: int,
        shares: dict
        ):
    
    shares_calc = {key: (value if value is not None else 0) for key, value in shares.items()}
    total_provided_share = sum(shares_calc.values())

    if total_provided_share > 100:
        raise ValueError('Error: Shares exceed 100%')

    if shares.get('g') is None:
        g_share = {'g': 100 - total_provided_share}
        shares.update(g_share)
    
    
    # Calculate the number of customers per cluster
    cluster_sizes = {key: int(cohort_size * (share / 100)) for key, share in shares.items()}
    
    # Handle rounding errors to ensure the total matches cohort_size
    total_assigned = sum(cluster_sizes.values())
    if total_assigned < cohort_size:
        # Add the remainder to the cluster g ('free to play')
        # largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        cluster_sizes['g'] += cohort_size - total_assigned
    
    # Create the customers dictionary
    users = {}
    start_index = 0
    for cluster, size in cluster_sizes.items():
        for i in range(start_index, start_index + size):
            users[i] = cluster
        start_index += size
    
    return users


def plot_hourly_distribution(df, timestamp_column='timestamp'):
    """
    Plot a line graph showing hourly aggregated counts of timestamps.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing timestamps.
        timestamp_column (str): Column name containing the timestamps.
    
    Returns:
        None: Displays the plot.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Aggregate counts by hour
    hourly_counts = df[timestamp_column].dt.floor('h').value_counts().sort_index()
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-', linewidth=2)
    plt.title('Hourly Distribution of Timestamps', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Count of Timestamps', fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plotly_hourly_distribution(df, timestamp_column='timestamp'):
    """
    Create an interactive Plotly line graph showing hourly aggregated counts of timestamps.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing timestamps.
        timestamp_column (str): Column name containing the timestamps.
    
    Returns:
        None: Displays the interactive plot.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Aggregate counts by hour
    hourly_aggregated = (
        df[timestamp_column]
        .dt.floor('h')  # Use 'h' (lowercase) for rounding to the nearest hour
        .value_counts()
        .sort_index()  # Ensure chronological order
        .reset_index()  # Convert index to a column
        .rename(columns={'timestamp': 'hour', 0: 'count'})  # Rename columns
    )
    hourly_aggregated = hourly_aggregated.reset_index()
#    hourly_aggregated = hourly_aggregated.rename(columns={'index': 'hour'})
    display(hourly_aggregated)
    
    # Add formatted columns for hover information
    hourly_aggregated['Date'] = hourly_aggregated['hour'].dt.date
    hourly_aggregated['Weekday'] = hourly_aggregated['hour'].dt.strftime('%A')
    hourly_aggregated['Time'] = hourly_aggregated['hour'].dt.time
    
    # Create a Plotly line plot
    fig = px.line(
        hourly_aggregated,
        x='hour',
        y='count',
        title='Hourly Distribution of Timestamps',
        labels={'hour': 'Hour', 'count': 'Installations'},
        hover_data={
            'hour': False,  # Exclude raw hour from hover
            'Date': True,
            'Weekday': True,
            'Time': True,
            'count': True,
        },
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=8))  # Add markers for hover points
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Count of Timestamps',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.show()


## Auxiliary functions

def find_lognormal_params(target_mode, target_range, tolerance=0.01):
    """
    Determine suitable mu and sigma for a lognormal distribution with the given mode and range.

    Parameters:
        target_mode (float): Desired mode of the distribution (in seconds).
        target_range (tuple): Desired range (in seconds) where most values should fall.
        tolerance (float): Allowed deviation from 68% probability within the range.

    Returns:
        tuple: (mu, sigma) for the lognormal distribution.
    """
    # Define range bounds
    lower_bound, upper_bound = target_range

    best_probability = 0

    # Iterate over possible sigma values
    for sigma in np.linspace(0.1, 1.0, 1000):
        # Calculate mu based on the mode equation
        mu = np.log(target_mode) + sigma**2
        
        # Calculate the CDF at the range bounds
        cdf_lower = lognorm.cdf(lower_bound, s=sigma, scale=np.exp(mu))
        cdf_upper = lognorm.cdf(upper_bound, s=sigma, scale=np.exp(mu))
        
        # Check if the cumulative probability within the range meets the tolerance

        new_probability = cdf_upper - cdf_lower

        if new_probability > best_probability:
            best_probability = new_probability

        if (new_probability) >= (1 - tolerance):
            return mu, sigma, new_probability

    raise ValueError(f"Suitable parameters not found within the given tolerance.  Best probability reached: {best_probability}")


def plot_session_start_distribution(
        initial_timestamp,
        mu,
        sigma,
        num_samples=1000,
        max_days=180,
        time_unit='seconds',
        second_mu=None,
        second_sigma=None,
        second_label='Second Distribution'
    ):
    """
    Plot the distribution of session start times based on one or two lognormal distributions.

    Parameters:
        initial_timestamp (str): Initial session start time in 'YYYY-MM-DD HH:MM:SS' format.
        mu (float): Mean (log scale) for the primary lognormal distribution.
        sigma (float): Standard deviation (log scale) for the primary lognormal distribution.
        num_samples (int): Number of samples to generate for the distributions.
        max_days (int): Maximum number of days after the initial timestamp to consider.
        time_unit (str): Unit for x-axis ('seconds', 'minutes', 'hours', 'days').
        second_mu (float, optional): Mean (log scale) for the secondary lognormal distribution.
        second_sigma (float, optional): Standard deviation (log scale) for the secondary lognormal distribution.
        second_label (str): Label for the secondary distribution.

    Returns:
        None: Displays the plot.
    """
    # Time unit conversion factors
    time_factors = {
        'seconds': 1,
        'minutes': 60,
        'hours': 3600,
        'days': 86400
    }

    if time_unit not in time_factors:
        raise ValueError("Invalid time_unit. Choose from 'seconds', 'minutes', 'hours', or 'days'.")

    # Convert initial timestamp to pandas Timestamp
    initial_time = pd.Timestamp(initial_timestamp)

    # Generate lognormal distribution for the primary distribution
    seconds_after = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)

    # Limit timescale to max_days
    max_seconds = max_days * 24 * 60 * 60  # Convert days to seconds
    seconds_after = seconds_after[seconds_after <= max_seconds]

    # Scale seconds_after based on the selected time_unit
    scaled_values = seconds_after / time_factors[time_unit]

    # Create a histogram for the primary distribution
    count, bins, _ = plt.hist(scaled_values, bins=100, density=True, alpha=0.5, color='blue', align='mid', label='Primary Distribution')

    # Generate PDF for the primary distribution
    x = np.linspace(min(bins), max(bins), 10000)
    pdf = (np.exp(-(np.log(x * time_factors[time_unit]) - mu)**2 / (2 * sigma**2)) /
           (x * time_factors[time_unit] * sigma * np.sqrt(2 * np.pi))) * time_factors[time_unit]
    plt.plot(x, pdf, linewidth=2, color='red', label='Primary PDF')

    # Optional: Generate and plot the second distribution
    if second_mu is not None and second_sigma is not None:
        # Generate lognormal distribution for the secondary distribution
        second_seconds_after = np.random.lognormal(mean=second_mu, sigma=second_sigma, size=num_samples)
        second_seconds_after = second_seconds_after[second_seconds_after <= max_seconds]
        second_scaled_values = second_seconds_after / time_factors[time_unit]

        # Create a histogram for the secondary distribution
        plt.hist(second_scaled_values, bins=100, density=True, alpha=0.5, color='green', align='mid', label=second_label)

        # Generate PDF for the secondary distribution
        second_pdf = (np.exp(-(np.log(x * time_factors[time_unit]) - second_mu)**2 / (2 * second_sigma**2)) /
                      (x * time_factors[time_unit] * second_sigma * np.sqrt(2 * np.pi))) * time_factors[time_unit]
        plt.plot(x, second_pdf, linewidth=2, color='orange', label=f'{second_label} PDF')

    # Plot settings
    plt.title('Session Start Time Distributions')
    plt.xlabel(f'Time After Initial Session ({time_unit})')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

    # Optional: Print the first few session times for inspection
#    print("First few session start times (Primary):")
#    primary_session_start_times = [initial_time + pd.Timedelta(seconds=sec) for sec in seconds_after]
#    print(pd.Series(primary_session_start_times).sort_values().head(10))

#    if second_mu is not None and second_sigma is not None:
#        print("\nFirst few session start times (Secondary):")
#        secondary_session_start_times = [initial_time + pd.Timedelta(seconds=sec) for sec in second_seconds_after]
#        print(pd.Series(secondary_session_start_times).sort_values().head(10))

    # Return expected value of that distribution
    return math.exp(mu + (sigma ** 2) / 2)



def plot_session_length_distribution(mu, sigma, num_samples=1000, max_seconds=86400, time_unit='minutes'):
    """
    Plot the distribution of session lengths based on a lognormal distribution.

    Parameters:
        mu (float): Mean (log scale) for the lognormal distribution.
        sigma (float): Standard deviation (log scale) for the lognormal distribution.
        num_samples (int): Number of samples to generate for the lognormal distribution.
        max_seconds (int): Maximum session length to consider (in seconds).
        time_unit (str): Unit for x-axis ('seconds', 'minutes', 'hours', 'days').

    Returns:
        None: Displays the plot.
    """
    # Time unit conversion factors
    time_factors = {
        'seconds': 1,
        'minutes': 60,
        'hours': 3600,
        'days': 86400
    }

    if time_unit not in time_factors:
        raise ValueError("Invalid time_unit. Choose from 'seconds', 'minutes', 'hours', or 'days'.")

    # Generate session lengths using the session_end function
    session_lengths = []
    for _ in range(num_samples):
        scaled_value, _ = session_end(mean=mu, std_dev=sigma)
        if scaled_value <= max_seconds:  # Ensure lengths are within max_seconds
            session_lengths.append(scaled_value)

    # Convert session lengths to the selected time unit
    scaled_values = np.array(session_lengths) / time_factors[time_unit]

    # Create a histogram for visualization
    count, bins, _ = plt.hist(scaled_values, bins=100, density=True, alpha=0.5, color='blue', align='mid')

    # Generate PDF for the bounded normal distribution
    x = np.linspace(min(bins), max(bins), 1000)
    pdf = (
        np.exp(-0.5 * ((x * time_factors[time_unit] - mu) / sigma)**2)
        / (sigma * np.sqrt(2 * np.pi))
    ) * time_factors[time_unit]
    plt.plot(x, pdf, linewidth=2, color='red', label='Bounded Normal PDF')

    # Plot settings
    plt.title('Session Length Distribution (Bounded Normal)')
    plt.xlabel(f'Session Length ({time_unit})')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

    # Optional: Print the first few session lengths for inspection
    print("First few session lengths:")
    print(pd.Series(scaled_values).sort_values().head(10))


def display_scrollable(df, max_rows=999):
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999
    html = df.to_html(max_rows=max_rows)
    html = f'<div style="max-height: 500px; overflow-y: scroll;">{html}</div>'
    display(HTML(html))
    pd.options.display.max_rows = 15
    pd.options.display.max_columns = 15


def simulate_churn_reactivation(churn_chance, reactivation_chance, sessions_per_day, num_customers, num_days=180):
    """
    Simulate churn and reactivation over a specified number of days for multiple customers.

    Parameters:
        churn_chance (float): Daily chance of a customer churning (as a percentage).
        reactivation_chance (float): Daily chance of a churned customer reactivating (as a percentage).
        sessions_per_day (float): Average number of sessions per day for active customers.
        num_customers (int): Number of simulated customers.
        num_days (int): Number of days to simulate (default is 180).

    Returns:
        pd.DataFrame: DataFrame with daily counts of active and churned users.
    """
    # Initialize results DataFrame
    results = pd.DataFrame(columns=['day', 'active_users', 'churned_users'])

    # Simulate each customer's status over time
    statuses = np.full(num_customers, 'active', dtype=object)  # Start all as active
    active_counts = []
    reactivated_counts = []
    churned_counts = []

    for day in range(1, num_days + 1):
        daily_active = 0
        daily_reactivated = 0
        daily_churned = 0

        for i in range(num_customers):
            if statuses[i] == 'active' or statuses[i] ==  'reactivated':
                # Generate sessions for the day
                num_sessions = np.random.poisson(sessions_per_day)
                for _ in range(num_sessions):
                    if np.random.random() < churn_chance / 100:
                        statuses[i] = 'churned'
                        break  # End sessions for this customer

            if statuses[i] == 'churned':
                if np.random.random() < reactivation_chance / 100:
                    statuses[i] = 'reactivated'

            # Update counts
            if statuses[i] == 'active':
                daily_active += 1
            elif statuses[i] == 'reactivated':
                daily_reactivated += 1
            else:
                daily_churned += 1

        active_counts.append(daily_active)
        reactivated_counts.append(daily_reactivated)
        churned_counts.append(daily_churned)

    # Populate results DataFrame
    results['day'] = range(1, num_days + 1)
    results['active_users'] = active_counts
    results['reactivated'] = reactivated_counts
    results['churned_users'] = churned_counts

    return results


def plot_churn_simulation(results):
    """
    Plot an area graph for the simulation results.

    Parameters:
        results (pd.DataFrame): DataFrame with daily counts of active and churned users.

    Returns:
        None: Displays the plot.
    """
    # Normalize to percentages
    total_users = results['active_users'] + results['reactivated'] + results['churned_users']
    results['active_share'] = results['active_users'] / total_users * 100
    results['reactivated_share'] = results['reactivated'] / total_users * 100
    results['churned_share'] = results['churned_users'] / total_users * 100

    # Plot the area graph
    plt.fill_between(results['day'], results['active_share'], label='Active Users', alpha=0.6)
    plt.fill_between(
        results['day'], 
        results['active_share'] + results['reactivated_share'], 
        results['active_share'], 
        label='Reactivated Users', 
        alpha=0.6
    )
    plt.fill_between(
        results['day'], 
        results['active_share'] + results['reactivated_share'] + results['churned_share'], 
        results['active_share'] + results['reactivated_share'], 
        label='Churned Users', 
        alpha=0.6
    )
    plt.title('Churn and Reactivation Simulation')
    plt.xlabel('Days')
    plt.ylabel('Percentage of Users')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ltv_over_time(df, time_column='timestamp', value_column='total_ltv_per_user'):
    """
    Create a line graph showing 'total_ltv_per_user' over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        time_column (str): Name of the column containing timestamps (or timedelta).
        value_column (str): Name of the column containing the LTV values.

    Returns:
        None: Displays the plot.
    """
    # Ensure the timestamp column is in datetime format
    #df[time_column] = pd.to_datetime(df[time_column])
    # Ensure the timedelta column is in timedelta format
    df[time_column] = pd.to_timedelta(df[time_column])
    
    # Format timedelta as "X days, Y hours"
    df['timedelta_formatted'] = df[time_column].apply(
        lambda x: f"{x.days} days, {x.components.hours} hours"
    )
    
    # Convert timedelta to total seconds for uniform x-axis scaling
    df['timedelta_seconds'] = df[time_column].dt.total_seconds()

    # Sort the DataFrame by timestamp to ensure proper time progression
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # Create a Plotly line graph
    fig = px.line(
        df,
        x='timedelta_formatted',#time_column,
        y=value_column,
        title='Total LTV Per User Over Time',
        labels={'timedelta_formatted': 'Time', value_column: 'Total LTV Per User'},
        hover_data={
#            timestamp_column: True,#: '|%Y-%m-%d %H:%M:%S',  # Format the timestamp in hover
            'timedelta_formatted': True,
            value_column: True  # Display the LTV value
        }
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=6))  # Add markers for better visibility
    fig.update_layout(
        xaxis_title='Time since install',
        yaxis_title='Total LTV Per User',
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Display the plot
    fig.show()


