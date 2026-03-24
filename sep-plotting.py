

def plot_scaling_factor_vs_mcp_radiation(radiation_dataset):
    ds = xr.open_dataset('radiation_data.nc')
    mcp_rads = ds["mcp_rad"].values
    scaling_factors = ds["scaling_factor"].values
    ds.close()

    # Linear regression analysis
    X = sm.add_constant(mcp_rads)  # Add intercept
    model = sm.OLS(scaling_factors, X).fit()
    print(model.summary())

    # Plot data with linear regression line
    plt.scatter(mcp_rads, scaling_factors, s=1, alpha=0.5, label='Data')
    plt.plot(mcp_rads, model.predict(X), 'r-', linewidth=2, label=f'Fit: y={model.params[1]:.4e}x+{model.params[0]:.4e}')
    plt.xlabel('MCP Radiation (W/m^2)')
    plt.ylabel('Scaling Factor')
    plt.title(f'MCP Radiation vs Scaling Factor\nR²={model.rsquared:.4f}')
    plt.legend()
    plt.savefig('scaling_factor_vs_mcp_radiation1.png', bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    main()