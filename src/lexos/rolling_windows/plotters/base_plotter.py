"""base_plotter.py.

Last Update: February 8, 2025
"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class BasePlotter(BaseModel):
    """BasePlotter class to enable type hinting and validation."""

    id: ClassVar[str] = "base_plotter"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    @property
    def metadata(self) -> dict:
        """Return metadata about the object.

        Returns:
            dict: A dictionary containing metadata about the object.
        """
        return self.model_dump()

    def _set_attrs(self, **kwargs) -> None:
        """Set instance attributes when public method is called.

        Args:
            attrs (dict): A dict of keyword arguments and their values.
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
