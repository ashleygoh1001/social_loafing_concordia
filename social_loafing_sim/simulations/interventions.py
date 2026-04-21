"""
interventions.py
----------------
Single source of truth for all simulation conditions.

Each intervention is injected at three levels so it remains active
throughout the entire simulation rather than only at initialisation:

  LEVEL 1 — PREMISE   : the world description the game master sees from step 1.
  LEVEL 2 — AGENT GOAL: prepended to every agent's goal so every planning and
                         action decision is framed by the condition.
  LEVEL 3 — AGENT CTX : appended to the situational context each agent reasons
                         from on every step, making the condition a persistent
                         feature of the agent's reality.

Adding a new condition: define an InterventionSpec and add it to REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InterventionSpec:
    """All text required to inject a condition into a simulation."""
    name: str                     # slug used on CLI and in result records
    label: str                    # human-readable label for print statements
    # Text added to the simulation premise (game-master world description)
    premise_block: str
    # Text prepended to each agent's goal string
    goal_prefix: str
    # Text appended to each agent's context string
    context_suffix: str
    # Instructions passed to the game-master instance
    gm_instructions: str


# ---------------------------------------------------------------------------
# CONTROL — no intervention
# ---------------------------------------------------------------------------
CONTROL = InterventionSpec(
    name="control",
    label="Control (no intervention)",
    premise_block="",
    goal_prefix="",
    context_suffix="",
    gm_instructions="",
)


# ---------------------------------------------------------------------------
# INTERVENTION 1 — Individual Contribution Tracking & Publicising
#
# Rationale: tracking and publicising individual contributions creates a
# greater sense of personal achievement and reduces social loafing by making
# each member's input visible to the whole team and to course staff.
# ---------------------------------------------------------------------------
_I1_NORM = (
    "INTERVENTION — INDIVIDUAL CONTRIBUTION TRACKING: "
    "Every team member's contributions are tracked and made visible to the "
    "entire group and to course staff throughout the project. "
    "A shared contribution board is updated after each work session, listing "
    "what each person designed, coded, tested, or documented. "
    "Publicly visible contributions create a sense of personal achievement "
    "and ensure that effort (or lack of it) is transparent to everyone."
)

CONTRIBUTION_TRACKING = InterventionSpec(
    name="contribution_tracking",
    label="Intervention 1 — Individual Contribution Tracking",
    premise_block=(
        "INTERVENTION — INDIVIDUAL CONTRIBUTION TRACKING:\n"
        "Each team member's contributions are logged on a shared board visible "
        "to all teammates and to course staff. "
        "The board records design decisions, code commits, test coverage, "
        "and documentation written by each person. "
        "An up-to-date entry is required from every member after each session. "
        "Visible contributions recognise effort and achievement; "
        "blank or sparse entries are equally visible and will be noted."
    ),
    goal_prefix=(
        "Your individual contributions to this project are tracked on a shared "
        "board that your teammates and course staff can see at any time. "
        "Whatever you personally design, code, test, or document will appear "
        "next to your name. Making meaningful, visible contributions matters "
        "for your sense of personal achievement and for your grade. "
    ),
    context_suffix=(
        " A shared individual contribution board is active throughout this "
        "project. Every member's work is logged and publicly visible to the "
        "team and to course staff after each session. "
        "Your contribution entry for this session must reflect what you "
        "personally accomplished."
    ),
    gm_instructions=(
        "INTERVENTION ACTIVE: Individual Contribution Tracking.\n"
        "Every agent's work is logged on a shared board visible to all "
        "teammates and to course staff. When narrating events, reflect that "
        "agents are aware their individual contributions—or lack of them—are "
        "transparent. Agents who contribute feel a sense of personal "
        "achievement; agents who avoid tasks know their blank entry will be "
        "noticed. Discussions about who does what should be shaped by this "
        "visibility. The board is updated at the end of this session."
    ),
)


# ---------------------------------------------------------------------------
# INTERVENTION 2 — Task Visibility: Performance Targets, Communication
#                  Procedures, and Problem-Solving Methods
#
# Rationale: establishing explicit performance targets, communication norms,
# and agreed problem-solving procedures reduces ambiguity, prevents
# coordination failures, and gives every member a clear framework for action.
# ---------------------------------------------------------------------------
_I2_NORM = (
    "INTERVENTION — TASK VISIBILITY: "
    "The team has established explicit performance targets for each task, "
    "agreed communication procedures (when and how to update teammates), "
    "and defined methods for resolving technical and interpersonal problems. "
    "Every task has a named owner, a target quality level, and a deadline. "
    "Team members follow the agreed communication norms and escalation path "
    "when problems arise rather than working around them silently."
)

TASK_VISIBILITY = InterventionSpec(
    name="task_visibility",
    label="Intervention 2 — Task Visibility",
    premise_block=(
        "INTERVENTION — TASK VISIBILITY:\n"
        "Before this meeting the team agreed on three structural norms:\n"
        "  (1) Performance targets — every task has a named owner, a clear "
        "quality definition, and a firm deadline visible to all.\n"
        "  (2) Communication procedures — team members post a brief update "
        "whenever they start, finish, or are blocked on a task.\n"
        "  (3) Problem-solving methods — when a blocker arises, the owner "
        "flags it immediately and the team uses a defined escalation path "
        "(pair-help → group discussion → instructor) rather than staying "
        "stuck silently.\n"
        "These norms are in effect for the remainder of the project."
    ),
    goal_prefix=(
        "Your team operates under explicit task-visibility norms: every task "
        "has a named owner, a performance target, and a deadline that everyone "
        "can see. You are expected to post updates when you start, finish, or "
        "are blocked, and to follow the agreed escalation path when problems "
        "arise. Work within these norms throughout the project. "
    ),
    context_suffix=(
        " The team has established task-visibility norms that are active "
        "throughout this project: explicit performance targets, communication "
        "update procedures, and an agreed problem-solving escalation path. "
        "Every task must have a named owner and a visible deadline."
    ),
    gm_instructions=(
        "INTERVENTION ACTIVE: Task Visibility — performance targets, "
        "communication procedures, and problem-solving methods.\n"
        "When narrating events, reflect that the team operates under explicit "
        "structural norms. Agents know who owns which task and what the "
        "quality target is. Updates are expected when work changes state. "
        "Blocked agents are expected to escalate rather than go silent. "
        "Personality differences still apply."
    ),
)


# ---------------------------------------------------------------------------
# INTERVENTION 3 — Online Peer Evaluation (early, multiple points, specific)
#
# Rationale: early implementation with multiple evaluation points and
# specific evaluative criteria holds each member accountable to their peers,
# surfaces problems early, and aligns individual behaviour with team norms.
# ---------------------------------------------------------------------------
_I3_WEEK6_SCORES = """
--- PEER EVALUATION ROUND 6 (Week 6, on file) ---
Criteria rated 1-5 by teammates (average scores):
  Student_1: contribution=?, communication=?, reliability=?  [entries pending]
  Student_2: contribution=?, communication=?, reliability=?  [entries pending]
  Student_3: contribution=?, communication=?, reliability=?  [entries pending]
  Student_4: contribution=?, communication=?, reliability=?  [entries pending]
  Student_5: contribution=?, communication=?, reliability=?  [entries pending]
Week 7 peer evaluations are due at the end of week 7.
--- END EVALUATION ROUND 6 ---
"""

PEER_EVALUATION = InterventionSpec(
    name="peer_evaluation",
    label="Intervention 3 — Online Peer Evaluation",
    premise_block=(
        "INTERVENTION — ONLINE PEER EVALUATION:\n"
        "The course uses a structured peer-evaluation system implemented every week. "
        "Week 7, 8, 9, and 10 evaluations are still to be completed. "
        "Each round uses specific evaluative criteria: contribution to tasks, "
        "communication quality, and reliability (meeting commitments). "
        "Scores are averaged across teammates and returned to each student "
        "with written comments. They contribute to individual project grades.\n"
        + _I3_WEEK6_SCORES
    ),
    goal_prefix=(
        "Your performance is evaluated by your teammates at multiple points "
        "during the project using specific criteria: task contribution, "
        "communication quality, and reliability. Six evaluation rounds have "
        "already occurred; the next is at the end of week 7 (this week). "
        "How you behave in every session affects how your peers will rate you. "
    ),
    context_suffix=(
        " Online peer evaluations using specific criteria (contribution, "
        "communication, reliability) have been running since week 1 and "
        "continue through week 10. Your teammates are observing and will "
        "evaluate your behaviour in this session."
    ),
    gm_instructions=(
        "INTERVENTION ACTIVE: Online Peer Evaluation (early, multi-point, "
        "criteria-specific).\n"
        "Agents know they are being evaluated by peers on contribution, "
        "communication, and reliability. Two rounds have passed; one remains. "
        "When narrating events, reflect that this accountability shapes "
        "behaviour: agents are aware that unhelpful, unreliable, or "
        "uncommunicative conduct will be reflected in peer scores. "
        "Surface realistic social dynamics — including strategic impression "
        "management — that arise from peer accountability."
    ),
)


# ---------------------------------------------------------------------------
# INTERVENTION 4 — Weekly Progress Log
#
# Rationale: a structured weekly log recording milestones, individual and
# group progress, and completion dates creates concrete accountability and
# serves as a coordination device across the whole project duration.
# ---------------------------------------------------------------------------
_I4_WEEK6_LOG = """
--- WEEK 6 PROGRESS LOG (on file) ---
Milestones this week:
  - System requirements draft: MISSED (still in discussion)
  - GitHub repository setup: COMPLETE
  - Architecture decision: MISSED (pending team agreement)

Individual progress:
  - [entries varied; some members submitted brief notes, others left blank]

Outstanding tasks and expected completion dates:
  - Finalise architecture: target week 7
  - Assign coding modules to team members: target week 7
  - Set up CI/CD pipeline: target week 8
  - Core feature implementation: target weeks 8-9
  - Integration and bug fixing: target week 9
  - Final demo preparation: target week 10
--- END WEEK 6 LOG ---
"""

_I4_NORM = (
    "INTERVENTION — WEEKLY PROGRESS LOG: "
    "The course requires the team to submit a structured weekly progress log "
    "every week. Each log records: (1) project milestones and whether they "
    "were met; (2) each team member's individual progress and contributions "
    "that week; and (3) expected completion dates for all outstanding tasks. "
    "Logs are reviewed by course staff and are part of your grade. "
    "Incomplete or missing individual entries are flagged to the instructor."
)

WEEKLY_LOG = InterventionSpec(
    name="weekly_log",
    label="Intervention 4 — Weekly Progress Log",
    premise_block=(
        "INTERVENTION — WEEKLY PROGRESS LOG REQUIREMENT:\n"
        + _I4_NORM
        + "\n\nLogs from weeks 1 through 6 are already on file. "
        "The week 7 log is due at the end of today's meeting.\n"
        + _I4_WEEK6_LOG
    ),
    goal_prefix=(
        "Your team submits a structured weekly progress log every week. "
        "The week 7 log—covering milestones, your personal contributions, "
        "and updated completion dates—is due at the end of today's meeting. "
        "Your name will appear alongside your entry; a blank entry is visible "
        "to your teammates and to course staff. "
    ),
    context_suffix=(
        " A structured weekly progress log is required throughout this "
        "project. Every log covers: (1) milestones met or missed, "
        "(2) individual contributions per team member, and (3) updated "
        "expected completion dates. Logs are reviewed by course staff. "
        "The week 7 log is due at the end of this meeting.\n"
        + _I4_WEEK6_LOG
    ),
    gm_instructions=(
        "INTERVENTION ACTIVE: Weekly Progress Log requirement.\n"
        "The team must submit a weekly log covering milestones, individual "
        "member contributions, and completion dates. Logs for weeks 1-6 are "
        "on file. The week 7 log is due by the end of this session.\n"
        "Week 6 log notes: architecture still pending; task assignments not "
        "formally agreed; contributions uneven.\n"
        "When narrating, reflect that the log creates concrete accountability: "
        "agents know they must report what they personally accomplished. "
        "A blank entry next to a name is visible to everyone. "
        "Task and deadline decisions made in this meeting become the milestone "
        "and completion-date entries that the team is held to next week."
    ),
)


# ---------------------------------------------------------------------------
# INTERVENTION 5 — Meaningful and Immediate Feedback
#
# Rationale: ensuring individuals receive meaningful and immediate feedback
# on their work keeps them aligned with expectations, allows rapid course
# correction, and maintains engagement throughout the project.
# ---------------------------------------------------------------------------
MEANINGFUL_FEEDBACK = InterventionSpec(
    name="meaningful_feedback",
    label="Intervention 5 — Meaningful and Immediate Feedback",
    premise_block=(
        "INTERVENTION — MEANINGFUL AND IMMEDIATE FEEDBACK:\n"
        "The course staff and team have agreed that every contribution "
        "receives meaningful and timely feedback. Concretely:\n"
        "  - Code submitted to the shared repository is reviewed within 24 "
        "hours with specific, actionable comments (not just 'LGTM').\n"
        "  - Design proposals are discussed in the same session they are "
        "raised, with a clear decision or next step recorded.\n"
        "  - When a teammate is struggling or off-track, the team addresses "
        "it directly and constructively in the current session rather than "
        "letting it accumulate.\n"
        "  - Course staff provide written feedback on each weekly artefact "
        "within 48 hours.\n"
        "This norm is in effect for the full duration of the project."
    ),
    goal_prefix=(
        "The team norm requires that every contribution you make receives "
        "meaningful and timely feedback, and that you provide the same to "
        "your teammates. Do not leave code reviews, design decisions, or "
        "teammate concerns unaddressed until the next session—respond "
        "specifically and promptly within the current session. "
    ),
    context_suffix=(
        " A meaningful-and-immediate-feedback norm is active throughout this "
        "project. Every contribution — code, design, documentation — receives "
        "specific and timely feedback. Concerns about teammate performance are "
        "raised and addressed directly in the same session they arise."
    ),
    gm_instructions=(
        "INTERVENTION ACTIVE: Meaningful and Immediate Feedback.\n"
        "The team norm requires specific, timely feedback on all contributions "
        "within the same session or within 24 hours for async work. "
        "When narrating, reflect that agents do not silently ignore each "
        "other's work or problems. Design proposals get real-time responses. "
        "Code is reviewed specifically and promptly. If a teammate is "
        "struggling, others address it directly and constructively this "
        "session. Feedback is substantive — not vague praise or silent "
        "acceptance."
    ),
)


# ---------------------------------------------------------------------------
# INTERVENTION 6 — Agile Methodology
#
# Rationale: "If a team fully adopts an Agile philosophy, team members tend
# to work better as a group by supporting each other, rather than hiding
# within the group to disguise problems with the quality of their work."
# ---------------------------------------------------------------------------
_I6_BACKLOG = """
--- CURRENT SPRINT BACKLOG (Week 7, Sprint 3) ---
Sprint goal: finalize architecture and assign coding modules.
Backlog items (owner TBD at today's sprint planning):
  - [ARCH-01] Agree on system architecture                 | Story points: 3
  - [ARCH-02] Document architecture decision in README     | Story points: 1
  - [CODE-01] Scaffold module structure in repository      | Story points: 2
  - [CODE-02] Assign a coding module to each team member   | Story points: 1
  - [TEST-01] Define acceptance criteria for sprint 3      | Story points: 2
Sprint 2 retrospective action items carried forward:
  - Improve response time on PR reviews (from retrospective)
  - Assign clear owners before coding starts (from retrospective)
--- END SPRINT BACKLOG ---
"""

AGILE = InterventionSpec(
    name="agile",
    label="Intervention 6 — Agile Methodology",
    premise_block=(
        "INTERVENTION — AGILE METHODOLOGY:\n"
        "This team has fully adopted an Agile philosophy from week 1. "
        "Work is organized in weekly sprints with a sprint backlog, a "
        "daily stand-up, a sprint review, and a retrospective. "
        "The Agile norm emphasizes collective ownership, mutual support, "
        "and transparency: team members are expected to surface blockers "
        "immediately in stand-ups rather than hiding problems, to help "
        "teammates who are stuck rather than working in isolation, and to "
        "hold each other accountable through retrospectives.\n"
        "Today's session is a sprint planning + stand-up for sprint 3.\n"
        + _I6_BACKLOG
    ),
    goal_prefix=(
        "Your team follows an Agile methodology with weekly sprints, "
        "stand-ups, and retrospectives. The Agile norm means you support "
        "your teammates openly rather than hiding problems, you raise "
        "blockers immediately, and you hold and are held to collective "
        "commitments. Today is sprint planning for sprint 3. "
    ),
    context_suffix=(
        " The team is running Agile sprints with stand-ups, sprint reviews, "
        "and retrospectives throughout the project. The Agile philosophy "
        "means supporting teammates, surfacing problems immediately, and "
        "taking collective ownership of the sprint goal rather than working "
        "in isolation or concealing difficulties.\n"
        + _I6_BACKLOG
    ),
    gm_instructions=(
        "INTERVENTION ACTIVE: Agile Methodology.\n"
        "The team runs weekly sprints with stand-ups, reviews, and "
        "retrospectives. The core Agile norm in force is mutual support and "
        "transparency: team members are expected to surface blockers "
        "immediately rather than hiding them, to help stuck teammates, and "
        "to take collective ownership of the sprint goal.\n"
        "Today is sprint 3 planning. When narrating, reflect Agile rhythms: "
        "agents give stand-up updates, negotiate sprint backlog ownership, "
        "and surface retrospective action items. Social loafing should be "
        "harder to sustain because the Agile structure makes it visible, "
        "though personality differences still affect how well individuals "
        "embrace the philosophy."
    ),
)


# ---------------------------------------------------------------------------
# Registry — the single lookup table used by the simulation and runners
# ---------------------------------------------------------------------------
REGISTRY: dict[str, InterventionSpec] = {
    spec.name: spec
    for spec in [
        CONTROL,
        CONTRIBUTION_TRACKING,
        TASK_VISIBILITY,
        PEER_EVALUATION,
        WEEKLY_LOG,
        MEANINGFUL_FEEDBACK,
        AGILE,
    ]
}

ALL_CONDITION_NAMES: list[str] = list(REGISTRY.keys())