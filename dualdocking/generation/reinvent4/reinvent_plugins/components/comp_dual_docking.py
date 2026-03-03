# Namespace-package plugin for REINVENT4 component discovery.
# Re-exports the plugin classes so REINVENT4 discovers them via pkgutil.walk_packages.
from dualdocking.generation.reinvent4.scoring import (  # noqa: F401
    DualDockingComponent,
    Parameters,
)
