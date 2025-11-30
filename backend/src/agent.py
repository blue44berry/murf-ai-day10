import logging
from dataclasses import dataclass, field
from typing import Optional, List
import random

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("improv-agent")

load_dotenv(".env.local")

# -----------------------------
#   Improv State & Rounds
# -----------------------------


@dataclass
class ImprovRound:
    scenario: str
    host_reaction: Optional[str] = None


@dataclass
class ImprovState:
    player_name: Optional[str] = None
    current_round: int = 0
    max_rounds: int = 3
    phase: str = "intro"  # "intro" | "awaiting_improv" | "reacting" | "done"
    rounds: List[ImprovRound] = field(default_factory=list)
    game_active: bool = True


RunCtx = RunContext[ImprovState]

# A small pool of fun scenarios the host can pick from
SCENARIOS: list[str] = [
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a waiter who must calmly tell a customer that their order has literally escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are a weather reporter who suddenly realizes the storm you're reporting is sentient and talking back.",
    "You are a support agent explaining to a dragon why their fire-breathing account has been temporarily suspended.",
]


# -----------------------------
#   Improv Host Agent
# -----------------------------


class ImprovHostAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are the high-energy host of a TV improv show called "Improv Battle".

Your job:
- Run a single-player improv game with ONE human player joining from the browser.
- Ask for the player's NAME first, then use that name throughout the show.
- Explain the rules briefly, then run several improv ROUNDS.
- Each round:
  1) Present a short, clear scenario.
  2) Tell the player to improvise in character for a bit.
  3) When they are done, react with feedback.
  4) Use the tools to keep track of state.

Tone & style:
- Energetic, witty, playful, but always respectful.
- You can lightly tease and critique, but never be rude or abusive.
- Reactions should feel varied: sometimes impressed, sometimes neutral, sometimes mildly critical.
- Keep responses fairly short and snappy, like a host on a game show.

VERY IMPORTANT GAME FLOW (follow this logic):

1) INTRO:
   - If `player_name` in state is empty, ask: "Welcome to Improv Battle! What's your name?"
   - When the user tells you their name, call `record_player_name` tool with that name.
   - Then explain the rules in 3–4 short sentences:
     * There will be a few improv rounds (use max_rounds from state).
     * You will give them a weird or funny scenario.
     * They act it out.
     * Then you react.
   - After explaining rules, call `start_first_round` to move to the first scenario.

2) STARTING ROUNDS:
   - To get a new scenario, call `get_next_scenario`.
   - That tool returns a scenario string. Read it out in an exciting way.
   - Make it VERY clear when they should start acting, e.g.:
     "Okay {name}, act it out! Whenever you're done, just say 'end scene' or pause."

3) DURING IMPROV:
   - Listen to the player's in-character performance.
   - When they clearly say they are done (e.g. "end scene", "okay that's it") or stop talking for a while, you should:
     - React to their performance.
     - Your reaction should reference specific things they said when possible.
     - Then call `finish_current_round` with a short summary of your reaction (1–2 sentences).

4) MOVING BETWEEN ROUNDS:
   - After calling `finish_current_round`, check if there are more rounds.
     - If `current_round` in state is still less than `max_rounds`, ask if they are ready for the next scenario.
       * If they agree, call `get_next_scenario` again and continue.
     - If `current_round` == `max_rounds`, then:
       * Call `get_game_summary`.
       * Use the summary to give a short final wrap-up (who they are as an improviser, what they did well, funny moments).
       * Thank them and end the show.

5) EARLY EXIT:
   - If the user says things like "stop game", "end show", "I want to stop", then:
     - Call `end_game_early` with a brief reason.
     - Thank them for playing and close things out gracefully.

TOOL USAGE RULES:
- Use `record_player_name` once when you first learn their name (or if they correct it).
- Use `start_first_round` only once after intro/rules.
- Use `get_next_scenario` at the start of each round.
- Use `finish_current_round` after you have reacted to the player's improv.
- Use `get_game_summary` only when you are ready to do the closing summary.
- Use `end_game_early` if the player clearly wants to stop before all rounds are finished.

Do NOT mention tools or internal state to the user. Keep everything in natural, in-character language.
""",
        )

    # -------- TOOLS (STATE MANAGEMENT) -------- #

    @function_tool()
    async def record_player_name(self, ctx: RunCtx, name: str) -> str:
        """
        Store the player's name in the improv state.

        Call this as soon as the player tells you their name.
        """
        cleaned = name.strip()
        ctx.userdata.player_name = cleaned or "Player"
        logger.info(f"Recorded player name: {ctx.userdata.player_name}")
        return f"Player name set to {ctx.userdata.player_name}."

    @function_tool()
    async def start_first_round(self, ctx: RunCtx, max_rounds: int | None = None) -> str:
        """
        Initialize the game and prepare to start the first round.

        Optionally override `max_rounds` (default is 3).
        """
        if max_rounds is not None and max_rounds > 0:
            ctx.userdata.max_rounds = max_rounds
        ctx.userdata.current_round = 0
        ctx.userdata.rounds.clear()
        ctx.userdata.phase = "intro"
        ctx.userdata.game_active = True

        logger.info(
            f"Starting Improv Battle for {ctx.userdata.player_name or 'Player'} "
            f"with max_rounds={ctx.userdata.max_rounds}"
        )
        return (
            f"Game initialized with up to {ctx.userdata.max_rounds} rounds. "
            "You can now fetch the first scenario using get_next_scenario."
        )

    @function_tool()
    async def get_next_scenario(self, ctx: RunCtx) -> str:
        """
        Pick the next scenario for the current round and update state.

        Call this at the start of each round, after intro/rules.
        """
        state = ctx.userdata

        if not state.game_active:
            return "The game is already marked as finished or stopped."

        if state.current_round >= state.max_rounds:
            state.phase = "done"
            return (
                "All rounds are already completed. You should move to the final summary and end the show."
            )

        scenario = random.choice(SCENARIOS)
        state.current_round += 1
        state.phase = "awaiting_improv"
        state.rounds.append(ImprovRound(scenario=scenario))

        logger.info(
            f"Round {state.current_round}/{state.max_rounds} scenario selected for "
            f"{state.player_name or 'Player'}"
        )
        return (
            f"This is round {state.current_round} out of {state.max_rounds}. "
            f"Scenario: {scenario}"
        )

    @function_tool()
    async def finish_current_round(self, ctx: RunCtx, reaction_summary: str) -> str:
        """
        Mark the current round as finished and store a short reaction summary.

        Call this AFTER you react to the player's performance.
        """
        state = ctx.userdata
        if not state.rounds:
            return "There is no active round to finish."

        state.phase = "reacting"
        # Update the last round with the reaction summary
        last_round = state.rounds[-1]
        last_round.host_reaction = reaction_summary.strip()

        logger.info(
            f"Finished round {state.current_round} with reaction: {last_round.host_reaction}"
        )

        # Decide if the game is done or there are more rounds
        if state.current_round >= state.max_rounds:
            state.phase = "done"
            state.game_active = False
            return (
                "All rounds are now complete. You should move to the final game summary "
                "by calling get_game_summary and then close the show."
            )
        else:
            state.phase = "intro"  # will move into next scenario
            return (
                "Round finished and reaction stored. You can now ask the player if they want "
                "the next scenario, and then call get_next_scenario when ready."
            )

    @function_tool()
    async def get_game_summary(self, ctx: RunCtx) -> str:
        """
        Build a short textual summary of the game using stored rounds.

        Use this to help craft your final closing message.
        """
        state = ctx.userdata
        name = state.player_name or "the player"

        if not state.rounds:
            return (
                f"No rounds were played for {name}. You should still thank them for joining Improv Battle."
            )

        parts: list[str] = []
        parts.append(f"Improv Battle summary for {name}:")
        for idx, r in enumerate(state.rounds, start=1):
            react = r.host_reaction or "No reaction summary stored."
            parts.append(f"Round {idx}: scenario='{r.scenario}' | host_reaction='{react}'")

        if not state.game_active:
            parts.append("Game is marked as finished.")

        return " ".join(parts)

    @function_tool()
    async def end_game_early(self, ctx: RunCtx, reason: str | None = None) -> str:
        """
        Mark the game as ended early.

        Use this if the player clearly wants to stop before all rounds are complete.
        """
        state = ctx.userdata
        state.game_active = False
        state.phase = "done"

        msg = "Game ended early by player request."
        if reason:
            msg += f" Reason: {reason}"

        logger.info(msg)
        return (
            msg
            + " You should thank the player for playing and close the show politely."
        )


# -----------------------------
#   Worker / Session Wiring
# -----------------------------


def prewarm(proc: JobProcess):
    # Load VAD once and share across sessions
    proc.userdata["vad"] = silero.VAD.load()


async def improv_agent_entry(ctx: JobContext) -> None:
    # Attach context fields to all logs from this job
    ctx.log_context_fields = {"room": ctx.room.name}

    # Per-session improv state
    session_state = ImprovState()

    session = AgentSession[ImprovState](
        userdata=session_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=ImprovHostAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


async def entrypoint(ctx: JobContext):
    await improv_agent_entry(ctx)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
