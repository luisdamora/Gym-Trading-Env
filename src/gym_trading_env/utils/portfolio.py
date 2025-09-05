class Portfolio:
    """Represent a trading portfolio with asset and fiat balances.

    Tracks spot holdings and their associated borrow interests. Provides helper
    methods to compute portfolio valuation and position, rebalance to a target
    position under trading fees, and update interests.
    """

    def __init__(self, asset, fiat, interest_asset=0, interest_fiat=0):
        """Initialize a portfolio.

        Args:
            asset (float): Quantity of the asset held (can be negative if short).
            fiat (float): Fiat balance (can be negative if borrowed).
            interest_asset (float): Accrued interest on borrowed asset. Optional.
            interest_fiat (float): Accrued interest on borrowed fiat. Optional.

        Returns:
            None: This constructor initializes instance attributes.
        """
        self.asset = asset
        self.fiat = fiat
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat

    def valorisation(self, price):
        """Compute the total portfolio value given a price.

        Args:
            price (float): Current price of the asset in fiat units.

        Returns:
            float: Portfolio valuation (asset*price + fiat - interests).
        """
        return sum(
            [self.asset * price, self.fiat, -self.interest_asset * price, -self.interest_fiat]
        )

    def real_position(self, price):
        """Compute the net exposure after subtracting borrow interests.

        Args:
            price (float): Current price of the asset.

        Returns:
            float: Net position as a fraction of portfolio value considering
                borrowed interest offsets.
        """
        return (self.asset - self.interest_asset) * price / self.valorisation(price)

    def position(self, price):
        """Compute the gross exposure (asset-only) fraction of portfolio value.

        Args:
            price (float): Current price of the asset.

        Returns:
            float: Position as a fraction of portfolio value from asset holdings only.
        """
        return self.asset * price / self.valorisation(price)

    def trade_to_position(self, position, price, trading_fees):
        """Trade to reach a target position given trading fees.

        First repays part of the borrow interests if moving closer to neutral
        from an over-extended short/long, then computes and executes the
        necessary trade size accounting for proportional trading fees.

        Args:
            position (float): Target position in [0, 1] for long-only or possibly
                outside if leverage/short is allowed by balances.
            price (float): Current asset price.
            trading_fees (float): Proportional fee rate (e.g., 0.001 for 10 bps).

        Returns:
            None: Updates internal balances in-place.
        """
        # Repay interest
        current_position = self.position(price)
        interest_reduction_ratio = 1
        if position <= 0 and current_position < 0:
            interest_reduction_ratio = min(1, position / current_position)
        elif position >= 1 and current_position > 1:
            interest_reduction_ratio = min(1, (position - 1) / (current_position - 1))
        if interest_reduction_ratio < 1:
            self.asset = self.asset - (1 - interest_reduction_ratio) * self.interest_asset
            self.fiat = self.fiat - (1 - interest_reduction_ratio) * self.interest_fiat
            self.interest_asset = interest_reduction_ratio * self.interest_asset
            self.interest_fiat = interest_reduction_ratio * self.interest_fiat

        # Proceed to trade
        asset_trade = position * self.valorisation(price) / price - self.asset
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + trading_fees * position)
            asset_fiat = -asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = -asset_trade * price
            self.asset = self.asset + asset_trade
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)

    def update_interest(self, borrow_interest_rate):
        """Update accrued interests based on current negative balances.

        Args:
            borrow_interest_rate (float): Interest rate applied to borrowed
                amounts for one period.

        Returns:
            None: Updates `interest_asset` and `interest_fiat` in-place.
        """
        self.interest_asset = max(0, -self.asset) * borrow_interest_rate
        self.interest_fiat = max(0, -self.fiat) * borrow_interest_rate

    def __str__(self):
        """Return a string representation of the portfolio state.

        Returns:
            str: Readable class name with current attributes.
        """
        return f"{self.__class__.__name__}({self.__dict__})"

    def describe(self, price):
        """Print the portfolio value and position for a given price.

        Args:
            price (float): Current asset price.

        Returns:
            None: Outputs to stdout.
        """
        print("Value : ", self.valorisation(price), "Position : ", self.position(price))

    def get_portfolio_distribution(self):
        """Get a non-negative breakdown of holdings and borrowings.

        Returns:
            dict: Keys include "asset", "fiat", "borrowed_asset", "borrowed_fiat",
                "interest_asset", and "interest_fiat".
        """
        return {
            "asset": max(0, self.asset),
            "fiat": max(0, self.fiat),
            "borrowed_asset": max(0, -self.asset),
            "borrowed_fiat": max(0, -self.fiat),
            "interest_asset": self.interest_asset,
            "interest_fiat": self.interest_fiat,
        }


class TargetPortfolio(Portfolio):
    """A portfolio instantiated from a target position and value.

    Computes the asset and fiat allocations such that the resulting portfolio
    matches the target position given the current price.
    """

    def __init__(self, position, value, price):
        """Initialize a target portfolio from desired position and total value.

        Args:
            position (float): Target fraction invested in the asset (0 to 1).
            value (float): Total portfolio valuation.
            price (float): Current asset price.

        Returns:
            None: Initializes the parent `Portfolio` with computed balances.
        """
        super().__init__(
            asset=position * value / price,
            fiat=(1 - position) * value,
            interest_asset=0,
            interest_fiat=0,
        )
