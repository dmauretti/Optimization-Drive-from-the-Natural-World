import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Read the convergence data
    df_cuckoo = pd.read_csv('cuckoo_convergence.csv')
    df_whale = pd.read_csv('whale_convergence.csv')
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Cuckoo Search vs Whale Optimization')
    
    # Best Cost Comparison
    ax.plot(df_cuckoo['iteration'], df_cuckoo['best_cost'], 
             'o-', label='Cuckoo Search', linewidth=2, markersize=4, color='#e74c3c')
    ax.plot(df_whale['iteration'], df_whale['best_cost'], 
             's-', label='Whale Optimization', linewidth=2, markersize=4, color='#3498db')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Best Cost', fontsize=11)
    ax.set_title('Best Cost Convergence', fontsize=12, fontweight='bold')
    ax.legend() 
    ax.grid(True, alpha=0.3)

    plt.savefig('convergence_comparison.png')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()