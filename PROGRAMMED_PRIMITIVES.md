# Programmed Primitives

This document strictly tracks what mechanics have been manually programmed into the RAI simulation, to ensure we can distinguish between programmed mechanics and emergent phenomena.

## Programmed Mechanics (v0.1 - Milestones 1-4)

1.  **Abstract Entities:** Discrete or continuous resources that can be possessed, transformed, and exchanged.
2.  **Scarcity:** Entities have finite quantities.
3.  **Transformations (Relations):** Hardcoded rules that consume input entities to produce output entities.
4.  **Knowledge:** Abstract tokens required to execute certain transformations.
5.  **Partial Observation:** Agents only observe a subset of the world state (their own inventory and local knowledge).
6.  **Exchange:** Agents can voluntarily swap entities.
7.  **Relation Discovery:** Agents can randomly discover new transformations and the knowledge required to execute them.
8.  **Knowledge Diffusion:** Agents can share knowledge with other agents.
9.  **Utility (Implicit):** Agents will eventually (via RL) seek to maximize a utility function based on their preferences for specific entities.
10. **Time (Ticks):** The simulation progresses in discrete time steps.

## Not Programmed (Emergence Candidates)

The following concepts are explicitly **NOT** programmed and their appearance would be considered emergent:

-   Currency / Medium of Exchange
-   Banks / Financial Institutions
-   Companies / Firms / Organizations
-   Markets
-   Investment
-   Credit / Loans
-   Specialization (Division of Labor)
-   Macroeconomic cycles (Recessions, Booms)
