import numpy as np


class History:
    """Store and query time-ordered records with flexible columns.

    The container flattens list and dict inputs into column names on first
    initialization via `set()`, then enforces the same schema for all
    subsequent `add()` calls. Access supports multiple forms via
    `__getitem__`: by time index, by column name, a tuple of (column, t), or a
    list of columns.
    """

    def __init__(self, max_size=10000):
        """Initialize the history buffer.

        Args:
            max_size (int): Maximum number of time steps to retain.
        """
        self.height = max_size

    def set(self, **kwargs):
        """Define the schema from provided fields and reset storage.

        Flattens lists and dicts into dedicated columns and builds an internal
        storage array with fixed width. Also performs an initial `add()` with
        the provided values.

        Args:
            **kwargs: Named fields for the initial record. Values can be
                scalars, lists, or dicts. Lists and dicts are flattened to
                multiple columns with suffixed names.

        Returns:
            None: This method does not return a value.
        """
        # Flattening the inputs to put it in np.array
        self.columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                self.columns.extend([f"{name}_{i}" for i in range(len(value))])
            elif isinstance(value, dict):
                self.columns.extend([f"{name}_{key}" for key in value.keys()])
            else:
                self.columns.append(name)

        self.width = len(self.columns)
        self.history_storage = np.zeros(shape=(self.height, self.width), dtype="O")
        self.size = 0
        self.add(**kwargs)

    def add(self, **kwargs):
        """Append a new record matching the initialized schema.

        Args:
            **kwargs: Named fields for the new record. Must match the schema
                defined by the initial `set()` call after flattening.

        Returns:
            None: This method does not return a value.

        Raises:
            ValueError: If the provided fields do not match the initialized
                columns.
        """
        values = []
        columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                columns.extend([f"{name}_{i}" for i in range(len(value))])
                values.extend(value[:])
            elif isinstance(value, dict):
                columns.extend([f"{name}_{key}" for key in value.keys()])
                values.extend(list(value.values()))
            else:
                columns.append(name)
                values.append(value)

        if columns == self.columns:
            self.history_storage[self.size, :] = values
            self.size = min(self.size + 1, self.height)
        else:
            raise ValueError(
                f"Make sur that your inputs match the initial ones... Initial ones : "
                f"{self.columns}. "
                f"New ones {columns}"
            )

    def __len__(self):
        """Return the number of stored records.

        Returns:
            int: Number of valid time steps currently stored.
        """
        return self.size

    def __getitem__(self, arg):
        """Retrieve records by time index, column, or combinations.

        Supports the following forms:
        - (column: str, t: int) -> scalar; value at time t for the column.
        - t: int -> dict; mapping column name to value at time t.
        - column: str -> numpy.ndarray; values for that column over stored rows.
        - columns: list[str] -> numpy.ndarray; table with the selected columns.

        Args:
            arg (tuple | int | str | list[str]): Indexing argument.

        Returns:
            object: Value or array depending on the input form.

        Raises:
            ValueError: If a requested column is not found.
        """
        if isinstance(arg, tuple):
            column, t = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as err:
                raise ValueError(
                    f"Feature {column} does not exist ... Check the available features : "
                    f"{self.columns}"
                ) from err
            return self.history_storage[: self.size][t, column_index]
        if isinstance(arg, int):
            t = arg
            return dict(zip(self.columns, self.history_storage[: self.size][t]))
        if isinstance(arg, str):
            column = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as err:
                raise ValueError(
                    f"Feature {column} does not exist ... Check the available features : "
                    f"{self.columns}"
                ) from err
            return self.history_storage[: self.size][:, column_index]
        if isinstance(arg, list):
            columns = arg
            column_indexes = []
            for column in columns:
                try:
                    column_indexes.append(self.columns.index(column))
                except ValueError as err:
                    raise ValueError(
                        f"Feature {column} does not exist ... Check the available features : "
                        f"{self.columns}"
                    ) from err
            return self.history_storage[: self.size][:, column_indexes]

    def __setitem__(self, arg, value):
        """Set a specific (column, t) cell value.

        Args:
            arg (tuple[str, int]): A pair of column name and time index.
            value (object): The value to assign.

        Returns:
            None: This method does not return a value.

        Raises:
            ValueError: If the column does not exist.
        """
        column, t = arg
        try:
            column_index = self.columns.index(column)
        except ValueError as err:
            raise ValueError(
                f"Feature {column} does not exist ... Check the available features : {self.columns}"
            ) from err
        self.history_storage[: self.size][t, column_index] = value
