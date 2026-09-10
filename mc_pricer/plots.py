"""Diagnostic plots: path fans, exercise boundaries, and P&L distributions."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional


def plot_path_fan(calendar: np.ndarray, paths: np.ndarray, div_schedule: Dict=None, strike: float=None, n_paths_display: int=100, save_path: Optional[str]=None, ticker: str='Stock', expiration_date: str=None, option_type: str='Call'):
    """Plot fan chart of simulated price paths with interactive hover."""
    dates = [d.astype('M8[D]').astype('O') for d in calendar]
    n_total = paths.shape[0]
    if n_total > n_paths_display:
        indices = np.random.choice(n_total, n_paths_display, replace=False)
        paths_display = paths[indices, :]
    else:
        paths_display = paths
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual paths
    for i in range(len(paths_display)):
        ax.plot(dates, paths_display[i, :], 'b-', alpha=0.1, linewidth=0.5)
    
    # Calculate and plot mean and percentiles
    mean_path = np.mean(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    
    line_mean, = ax.plot(dates, mean_path, 'r-', linewidth=2, label='Mean Path')
    line_p10, = ax.plot(dates, p10, 'g--', linewidth=1.5, label='10th Percentile')
    line_p90, = ax.plot(dates, p90, 'g--', linewidth=1.5, label='90th Percentile')
    
    if strike is not None:
        ax.axhline(y=strike, color='orange', linestyle='--', linewidth=2, label=f'Strike: ${strike:.2f}', alpha=0.7)
    
    # Plot dividend lines with purple color (distinct from green percentile lines)
    if div_schedule is not None and len(div_schedule['div_idx']) > 0:
        div_marked = False
        for idx, amount in zip(div_schedule['div_idx'], div_schedule['div_amounts']):
            if idx < len(dates):
                div_date = dates[idx]
                label = 'Dividend' if not div_marked else None
                ax.axvline(x=div_date, color='purple', linestyle='--', alpha=0.6, linewidth=2, label=label)
                ax.text(div_date, ax.get_ylim()[1] * 0.98, f'Dividend\n${amount:.2f}', rotation=0, horizontalalignment='center', verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='plum', alpha=0.7))
                div_marked = True
    
    # Set x-axis limit to expiration date (no whitespace after)
    ax.set_xlim(left=dates[0], right=dates[-1])
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    
    # Updated title format
    exp_date_str = expiration_date if expiration_date else dates[-1].strftime('%Y-%m-%d')
    option_label = option_type.capitalize()
    dte = (dates[-1] - dates[0]).days
    n_total_formatted = f'{n_total:,}'
    ax.set_title(f'{ticker} {option_label} Expiring {exp_date_str} Risk-Neutral Simulated Price Paths (showing {len(paths_display)} of {n_total_formatted}) DTE: {dte}', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add interactive hover annotation (only if using interactive backend)
    try:
        annot = ax.annotate('', xy=(0, 0), xytext=(20, 20), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9, edgecolor='black'),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5),
                            fontsize=9, zorder=100)
        annot.set_visible(False)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                # Find closest date index
                if event.xdata is not None:
                    try:
                        x_date = mdates.num2date(event.xdata)
                        distances = [abs((d - x_date.replace(tzinfo=None)).total_seconds()) for d in dates]
                        idx = np.argmin(distances)
                        
                        # Get values at that index
                        date_str = dates[idx].strftime('%Y-%m-%d')
                        mean_val = mean_path[idx]
                        p10_val = p10[idx]
                        p90_val = p90[idx]
                        
                        # Update annotation position and text
                        text = f'Date: {date_str}\nMean: ${mean_val:.2f}\n10th: ${p10_val:.2f}\n90th: ${p90_val:.2f}'
                        annot.xy = (mdates.date2num(dates[idx]), mean_val)
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    except:
                        pass
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', hover)
    except:
        # If hover functionality fails, continue without it
        pass
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_exercise_boundary(calendar: np.ndarray, exercise_boundary: np.ndarray, div_schedule: Dict, strike: float, save_path: Optional[str]=None, ticker: str='Stock', expiration_date: str=None, option_type: str='Call'):
    """Plot early exercise boundary over time."""
    dates = [d.astype('M8[D]').astype('O') for d in calendar]
    plt.figure(figsize=(12, 6))
    valid = ~np.isnan(exercise_boundary)
    valid_dates = [dates[i] for i in range(len(dates)) if valid[i]]
    valid_boundary = exercise_boundary[valid]
    if len(valid_dates) > 0:
        plt.plot(valid_dates, valid_boundary, 'b-', linewidth=2, label='Exercise Boundary')
    plt.axhline(y=strike, color='r', linestyle='--', linewidth=2, label=f'Strike: ${strike:.2f}')
    
    # Mark day before dividends with exercise boundary values
    div_marked = False
    if len(div_schedule['div_idx']) > 0:
        for idx, amount in zip(div_schedule['div_idx'], div_schedule['div_amounts']):
            if idx < len(dates) and idx > 0:  # Need idx > 0 to get day before
                day_before_div = dates[idx - 1]
                # Get exercise boundary value on day before dividend
                if idx - 1 < len(exercise_boundary) and not np.isnan(exercise_boundary[idx - 1]):
                    boundary_value = exercise_boundary[idx - 1]
                    label = 'Day Before Dividend' if not div_marked else None
                    plt.axvline(x=day_before_div, color='purple', linestyle='--', alpha=0.7, linewidth=2, label=label)
                    
                    # Add text annotation showing both dividend amount and exercise threshold
                    plt.text(day_before_div, plt.ylim()[1] * 0.98, 
                            f'Div: ${amount:.2f}\nExercise if S>${boundary_value:.2f}', 
                            rotation=0, horizontalalignment='center', verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='plum', alpha=0.8, edgecolor='purple'))
                    div_marked = True
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    
    # Updated title with ticker and date
    exp_date_str = expiration_date if expiration_date else 'N/A'
    option_label = option_type.capitalize()
    plt.title(f'{ticker} {option_label} Expiring {exp_date_str} - Early Exercise Boundary (Risk-Neutral)', fontsize=14, fontweight='bold')
    plt.suptitle('"At what stock price does early exercise become optimal at each point in time?"', 
                 fontsize=10, style='italic', y=0.96)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pnl_distribution(pnl_distribution: np.ndarray, premium_paid: float, save_path: Optional[str]=None, ticker: str='Stock', expiration_date: str=None, option_type: str='Call'):
    """Plot simple P&L distribution histogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(pnl_distribution, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Add reference lines
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
    ax.axvline(x=np.mean(pnl_distribution), color='green', linestyle='-', linewidth=2, 
               label=f'Expected P/L: ${np.mean(pnl_distribution):.2f}')
    
    ax.set_xlabel('Profit / Loss ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # Updated title with ticker and date
    exp_date_str = expiration_date if expiration_date else 'N/A'
    option_label = option_type.capitalize()
    ax.set_title(f'{ticker} {option_label} Expiring {exp_date_str} - P/L Distribution (Premium Paid: ${premium_paid:.2f})', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interactive hover annotation
    try:
        annot = ax.annotate('', xy=(0, 0), xytext=(20, 20), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9, edgecolor='black'),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5),
                            fontsize=9, zorder=100)
        annot.set_visible(False)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                # Check if mouse is over a histogram bar
                for i, patch in enumerate(patches):
                    cont, _ = patch.contains(event)
                    if cont:
                        try:
                            # Get bin edges and frequency
                            bin_left = bins[i]
                            bin_right = bins[i + 1]
                            bin_center = (bin_left + bin_right) / 2
                            frequency = int(n[i])
                            
                            # Update annotation
                            text = f'P/L: ${bin_center:.2f}\nFrequency: {frequency}\nRange: [${bin_left:.2f}, ${bin_right:.2f})'
                            annot.xy = (bin_center, n[i])
                            annot.set_text(text)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                        except:
                            pass
                        return
                
                # If not over any bar, hide annotation
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', hover)
    except:
        pass
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pnl_risk_analysis(pnl_distribution: np.ndarray, premium_paid: float, save_path: Optional[str]=None, ticker: str='Stock', expiration_date: str=None, option_type: str='Call'):
    """Plot P&L risk analysis with CDF and probability metrics."""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Sort P/L for CDF calculation
    pnl_sorted = np.sort(pnl_distribution)
    
    # Calculate key statistics
    prob_profit = np.mean(pnl_distribution > 0)
    var_5 = np.percentile(pnl_distribution, 5)  # VaR at 5% (worst 5% threshold)
    cvar_5 = np.mean(pnl_distribution[pnl_distribution < var_5])  # CVaR (expected shortfall - average of worst 5%)
    median_pnl = np.median(pnl_distribution)
    mean_pnl = np.mean(pnl_distribution)
    
    # Calculate and plot CDF on primary axis
    cdf_probs = np.arange(1, len(pnl_sorted) + 1) / len(pnl_sorted) * 100  # Convert to percentage
    ax1.plot(pnl_sorted, cdf_probs, color='darkblue', linewidth=3, 
             label='Cumulative Probability', alpha=0.8)
    
    ax1.set_xlabel('Profit / Loss ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability of Ending With ≤ X P/L (%)', fontsize=12, fontweight='bold', color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add reference lines with statistics
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Break Even', alpha=0.8)
    ax1.axvline(x=mean_pnl, color='green', linestyle='-', linewidth=2, 
                label=f'Expected P/L: ${mean_pnl:.2f}', alpha=0.8)
    ax1.axvline(x=median_pnl, color='orange', linestyle='-.', linewidth=2, 
                label=f'Median: ${median_pnl:.2f}', alpha=0.8)
    
    # Add statistics text box
    stats_text = (
        f'Risk Metrics:\n'
        f'─────────────────────\n'
        f'Prob(Profit): {prob_profit:.1%}\n'
        f'Prob(Loss): {(1-prob_profit):.1%}\n'
        f'─────────────────────\n'
        f'Expected P/L: ${mean_pnl:.2f}\n'
        f'Median P/L: ${median_pnl:.2f}'
    )
    
    # Position text box in upper left
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black'))
    
    # Title with ticker and date
    exp_date_str = expiration_date if expiration_date else 'N/A'
    option_label = option_type.capitalize()
    ax1.set_title(f'{ticker} {option_label} Expiring {exp_date_str} - Risk Analysis & Probability of Profit (Risk-Neutral)', 
                  fontsize=14, fontweight='bold', pad=15)
    
    ax1.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_option_value_over_time(calendar: np.ndarray, cashflows: np.ndarray, exercise_matrix: np.ndarray, 
                                 rate_curve: Dict, strike: float, option_type: str, save_path: Optional[str]=None, 
                                 ticker: str='Stock', expiration_date: str=None):
    """Plot discounted option value over time (Q-measure).
    
    Shows how the expected option value decays over time under risk-neutral pricing.
    Explains theta intuitively and shows time-value erosion.
    
    Args:
        calendar: Trading calendar
        cashflows: Cash flow matrix from LSM (n_paths x n_steps)
        exercise_matrix: Boolean exercise decision matrix (n_paths x n_steps)
        rate_curve: Discount curve
        strike: Strike price
        option_type: 'call' or 'put'
        save_path: Optional path to save figure
        ticker: Stock ticker
        expiration_date: Expiration date string
    """
    dates = [d.astype('M8[D]').astype('O') for d in calendar]
    n_paths, n_steps = cashflows.shape
    df = rate_curve['discount_factors']
    
    # Calculate expected discounted option value at each time step
    expected_values = np.zeros(n_steps)
    value_std = np.zeros(n_steps)
    
    for t in range(n_steps):
        # For each path, calculate the discounted value from time t
        path_values_at_t = np.zeros(n_paths)
        
        for path_idx in range(n_paths):
            # Find when this path exercises (at or after time t)
            exercise_times = np.where(exercise_matrix[path_idx, t:])[0]
            if len(exercise_times) > 0:
                # Exercise at first exercise time after t
                t_ex = t + exercise_times[0]
                # Discount cashflow from exercise time back to time t
                path_values_at_t[path_idx] = cashflows[path_idx, t_ex] * df[t] / df[t_ex]
            else:
                # No exercise, value is 0
                path_values_at_t[path_idx] = 0.0
        
        expected_values[t] = np.mean(path_values_at_t)
        value_std[t] = np.std(path_values_at_t)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot expected value
    ax.plot(dates, expected_values, 'b-', linewidth=3, label='Expected Option Value (Q-measure)', alpha=0.8)
    
    # Add confidence band (±1 std error)
    std_error = value_std / np.sqrt(n_paths)
    upper_band = expected_values + std_error
    lower_band = expected_values - std_error
    ax.fill_between(dates, lower_band, upper_band, color='blue', alpha=0.2, label='±1 Std Error')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Discounted Option Value ($)', fontsize=12, fontweight='bold')
    ax.set_xlim(left=dates[0], right=dates[-1])
    ax.grid(True, alpha=0.3)
    
    # Title
    exp_date_str = expiration_date if expiration_date else dates[-1].strftime('%Y-%m-%d')
    option_label = option_type.capitalize()
    ax.set_title(f'{ticker} {option_label} Expiring {exp_date_str} - Discounted Option Value vs Time (Risk-Neutral)', 
                 fontsize=14, fontweight='bold')
    plt.suptitle('Extrinsic (time) value will erode as expiry approaches', 
                 fontsize=10, style='italic', y=0.96)
    
    # Add theta annotation
    if len(dates) > 1:
        initial_value = expected_values[0]
        final_value = expected_values[-1]
        total_decay = initial_value - final_value
        days = (dates[-1] - dates[0]).days
        avg_theta_per_day = total_decay / days if days > 0 else 0
        
        stats_text = (
            f'Time Decay (Theta):\n'
            f'─────────────────────\n'
            f'Initial Value: ${initial_value:.2f}\n'
            f'Final Value: ${final_value:.2f}\n'
            f'Total Decay: ${total_decay:.2f}\n'
            f'Avg Theta/Day: ${avg_theta_per_day:.3f}'
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black'))
    
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
