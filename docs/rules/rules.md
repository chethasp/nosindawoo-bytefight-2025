# ByteFight 2025: Snake

Welcome to Georgia Tech's inaugural ByteFight competition! In the distant future, the digital world is overrun with with hyper-intelligent cyber-serpents. They hurtle across the neon landscape in a  scramble for domination, constrained only by their programmed protocols: Consume. Grow. Survive.


This is **ByteFight 2025: Snake**. 

## Getting Started
**Download our client:** See bytefight.org and find the client that matches your distribution.

## Setting up an IDE
For the inaugural season of ByteFight we will only be supporting Python, so we highly recommend installing **VS Code** or **PyCharm**. You do not need to install any python packages since they come packaged with the client. Instead, simply go to bytefight.org and download the scaffold, which contains the libraries you will use to read and interact with the game, documentation for those libraries, and the sample player.

## General Game Rules
In **ByteFight 2025: Snake**, you will outmaneuver, outwit, and outlast your opponents to claim victory in our online arena.

**How to win the game:** 
* Your opponent takes an *invalid actions*
* Your opponent's total length is less than the minimum size (2) at any point during the game
* Your opponent runs out of time
* Your opponent's program crashes
* Win by tiebreak

An invalid action consists of any action that decreases a snake's length below 2 or moves it into a wall or part of a snake.

**Tiebreak:**
Tiebreaks occur if the turn count reaches 2000 and is determined in the following order:

* Whichever team has eaten more apples
* Whichever team has the snake with the longer length at the end
* Whichever team has more time left on their clock (within a 5 second margin of error).
* If the above tiebreaks both don't work, the game is a tie.


## Actions
* **Action** - Either performing a *move*, placing a *trap*.
* **Move** - A move is a single cardinal or intercardinal direction, represented by an enum. See `game.enums.Action` for more.
* **Length** - How long a snake is, including length that has been queued but not actualized on the map. Is also a `game.enums.Action`.
* **Turn** - An entire turn is a single action or a series of actions (in the form of a list or array), consisting of at least one move . 

**Moves:**  
Snakes must move at least once on their turn. Snakes can move in directions that are at most 90 degrees away from the direction they are currently facing. Moving in a direction will face you in that direction. For example if a snake is currently facing north, on its next move it can move west, northwest, north, northeast, and east. Players can move in any direction if they have not taken a move yet during the game.

The first move a snake takes on its turn, it will first lose 1 length first at its tail, then gain 1 length at the new head position, preserving its total length. Every subsequent move during the turn will increase the sacrifice **+2**. So, the first move on a turn will sacrifice a total of zero length. The next move on the turn will sacrifice 3 length at the tail and gain 1 at the head, for a total of 2 length lost. The next move on the turn will sacrifice 5 length at the tail and gain 1 length at the head, for a total of 4 length lost. 

**First Move:**  
Players will bid length at the beginning of the game to have the first turn. Whoever bids higher will have the bid applied to them (they will lose length equal to how much they bid). Nothing will happen for the person who loses the bid. If both players bid the same amount, the person who wins the bid is decided by a coin toss.

**Traps:**  
A snake can sacrifice one of its unqueued length as an action to leave behind a trap at its current tail. Only unqueued length beyond 2 may be used to deploy a trap. For example, if your snake occupies two physical squares but had 4 length queued, you would not be able to leave a trap. Traps last on a tile for 50 turns following placement.

The total number of traps that you are allowed to place on a single turn is limited to half the max length you have achieved (including both unqueued and queued length), rounded down.

In the event that the opposing snake moves onto a square containing one of your traps, they will lose 2 length. In the event that your snake's head moves onto a square containing one of your traps, the trap's lifetime will be refreshed and it will last another 50 rounds.

## The Map
Maps will be either vertically, horizontally, or rotationally symmetric. Map dimensions will be, at smallest, 8x8 and at largest 64x64. A set number of apples will spawn at set turn intervals (10-200, always odd). Where and when apples will spawn will be known to players at the beginning of the game.

Cells may have any of the following:
* **Wall** (An impassable tile that snakes cannot move into. Cannot contain anything else.)
* **Apple** (A set will spawn every 10-200 turns)
* **Snake** 
* **Trap** 
* **Portal**


**Apples:**  
Whenever the snake head moves onto a cell with an apple, the snake will gain 2 length. This two length will be "queued" and will accrue on the map over the next 2 moves made by the player. While this length is accruing on the map, the player's tail will not move. Any "queued" length is still counted as part of the snake's length and will be sacrificed first in the event a player makes multiple moves on their turn or triggers a trap.


**Portals:**  
Whenever the snake head moves onto a cell with a portal, they will pass through warped space and time to emerge out their respective symmetrically-reflected portal. Portals technically exist as the "same cell" in space. If a snake occupies one end of the portal, they also occupy the other end. The same goes for apples and traps: the same apple and trap exist at each end of the portal, so you can interact with them by stepping through either end of the portal.

**Decay:**   
After 1000 turns, snakes become infected with the logic plague, causing them to **decay**. Snakes will lose 1 length every 15 turns at the beginning of their turn. At 1600 turns, this rate increases to every 10 turns. At 1800 turns, this rate increases to every 5 turns. At 1950 turns, this rate increases to every turn.

## How to Play
**Playing the game:**  
Players will be given a `player_board` instance, a copy of the main board through which they can visualize and interact with the game. On your turn, you will either `bid`
or `play` as described by the function in your `controller.py` class. A `bid` should return a valid integer, and a `play` should return a valid turn, comprised of actions. Note that returning `enums.Action.FF` as any part of your turn will cause you to forfeit the game. Do not use `enums.Action.FF` when calling player functions as it may lead to undefined behavior.

You can use any of the classes in the `game` module such as `game_queue`, `snake`, `board`, or `enums`. Be careful, though - you can safely read any of the variables of the classes, but we recommend you use our methods to mutate the boards. This is because accidentally breaking references of classes to their variables may lead to unexpected behavior. See our documentation for more.

**Resource Limitations:**  
In ranked, tournament, and scrimmage matches, your team will be allotted a total of **5 seconds** to construct your player class, **5 seconds** to bid, and **110 seconds** for all your moves following your bid, measured as the amount of time your main user process remains on the CPU before returning a move. Timing is done in chess-clock format: the moment your turn ends, your opponent's time starts ticking. You can retrieve how much time you have left via a function passed to your player, and how much time your opponent has left via retrieving it from `player_board`.

You will be guarenteed 90-99% of cpu time on 3 logical cores of a hyperthreaded Core i7-10750H. Your controller will be allotted at most 1GB of RAM - exceeding this amount will throw a `MemoryError`. For reference, the representation of a single board state is at maximum 80 KB of data. At the end of your turn, your player processes along with any subprocesses you created will be paused to give your opponent full compute time, so make sure calculation on all threads/processes you create are complete/pausable at the time of creation.

You will be provided a local client to develop, test, and debug your code. This local client, for the most part, does not take into account resource limitations that will apply to you during the final competition. The local client measures time used based on wall clock time, and provides you an abundance of time for your turns to execute, which will not be the case for the games on our server. We suggest that if you want to test time-based performance limitations, you first ensure your code works, then submit scrimmages against yourself to our servers (note that these scrimmages are rate limited).

We recommend if you want to utilize code parallelization to use python's `multiprocessing`, or use `numpy` vectorization. You can also use `numba` and `Cython`, although these two libraries are less well-supported and devs will out work out issues on a case-by-case basis (reach out to us if you have issues with them). Any dynamically linked libraries you create with `Cython` will need to be for Debian Ubuntu Linux since that is what we will run your submissions on. You are allowed to open, read, and load files during runtime, however you will not be able to delete, remove, or create files or directories during runtime.

**A note about the rules:**  
Note that all rules are subject to change for game balancing. These will generally be numbers tweaks, no new mechanics will be added to the game.

**Libraries:**  
For this first competition we are only allowing access to python standard libraries, `numba`,  and `numpy`. If you want another library feel free to submit a request, and the development team will consider adding it to the next iteration of the engine.

