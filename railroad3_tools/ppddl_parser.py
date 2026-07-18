from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union


@dataclass(frozen=True)
class TypedSymbol:
    name: str
    type_name: str = "object"


@dataclass(frozen=True)
class Literal:
    predicate: str
    arguments: Tuple[str, ...]
    positive: bool = True

    def ground(self, binding: dict[str, str]) -> "Literal":
        return Literal(
            self.predicate,
            tuple(binding.get(arg, arg) for arg in self.arguments),
            self.positive,
        )

    def atom(self) -> Tuple[str, Tuple[str, ...]]:
        return self.predicate, self.arguments


@dataclass(frozen=True)
class Equality:
    left: str
    right: str
    equal: bool = True


Condition = Union[Literal, Equality]


@dataclass
class Outcome:
    probability: float
    effects: List[Literal]


@dataclass
class ActionSchema:
    name: str
    parameters: List[TypedSymbol]
    preconditions: List[Condition]
    outcomes: List[Outcome]


@dataclass
class PredicateDecl:
    name: str
    parameters: List[TypedSymbol]


@dataclass
class Domain:
    name: str
    requirements: List[str] = field(default_factory=list)
    predicates: List[PredicateDecl] = field(default_factory=list)
    actions: List[ActionSchema] = field(default_factory=list)


@dataclass
class Problem:
    name: str
    domain_name: str
    objects: List[TypedSymbol]
    initial_literals: List[Literal]
    goal_conditions: List[Condition]


SExpr = Union[str, List["SExpr"]]


def tokenize(text: str) -> List[str]:
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(text: str) -> SExpr:
    tokens = tokenize(text)
    root: List[SExpr] = []
    stack: List[List[SExpr]] = []
    current = root

    for token in tokens:
        if token == "(":
            node: List[SExpr] = []
            current.append(node)
            stack.append(current)
            current = node
        elif token == ")":
            if not stack:
                raise ValueError("Unexpected ')' in PDDL")
            current = stack.pop()
        else:
            current.append(token)

    if stack:
        raise ValueError("Unclosed '(' in PDDL")
    if len(root) != 1:
        raise ValueError(f"Expected one top-level form, found {len(root)}")
    return root[0]


def _as_list(value: SExpr, context: str) -> List[SExpr]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {context}, got {value!r}")
    return value


def _typed_symbols(items: Sequence[SExpr]) -> List[TypedSymbol]:
    tokens = [str(x) for x in items]
    result: List[TypedSymbol] = []
    pending: List[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token == "-":
            if i + 1 >= len(tokens):
                raise ValueError("Missing type after '-'")
            type_name = tokens[i + 1]
            result.extend(TypedSymbol(name, type_name) for name in pending)
            pending.clear()
            i += 2
            continue
        pending.append(token)
        i += 1

    result.extend(TypedSymbol(name, "object") for name in pending)
    return result


def _flatten_condition(expr: SExpr) -> List[Condition]:
    form = _as_list(expr, "condition")
    if not form:
        return []

    head = str(form[0]).lower()
    if head == "and":
        result: List[Condition] = []
        for part in form[1:]:
            result.extend(_flatten_condition(part))
        return result

    if head == "not":
        inner = _as_list(form[1], "negated condition")
        if inner and str(inner[0]) == "=":
            return [Equality(str(inner[1]), str(inner[2]), equal=False)]
        return [
            Literal(
                predicate=str(inner[0]),
                arguments=tuple(str(x) for x in inner[1:]),
                positive=False,
            )
        ]

    if head == "=":
        return [Equality(str(form[1]), str(form[2]), equal=True)]

    return [
        Literal(
            predicate=str(form[0]),
            arguments=tuple(str(x) for x in form[1:]),
            positive=True,
        )
    ]


def _flatten_effect(expr: SExpr) -> List[Literal]:
    form = _as_list(expr, "effect")
    if not form:
        return []

    head = str(form[0]).lower()
    if head == "and":
        result: List[Literal] = []
        for part in form[1:]:
            result.extend(_flatten_effect(part))
        return result

    if head == "not":
        inner = _as_list(form[1], "negated effect")
        return [
            Literal(
                predicate=str(inner[0]),
                arguments=tuple(str(x) for x in inner[1:]),
                positive=False,
            )
        ]

    return [
        Literal(
            predicate=str(form[0]),
            arguments=tuple(str(x) for x in form[1:]),
            positive=True,
        )
    ]


def _parse_outcomes(effect_expr: SExpr) -> List[Outcome]:
    form = _as_list(effect_expr, "action effect")
    if form and str(form[0]).lower() == "probabilistic":
        if (len(form) - 1) % 2 != 0:
            raise ValueError("Malformed probabilistic effect")
        outcomes: List[Outcome] = []
        for i in range(1, len(form), 2):
            probability = float(str(form[i]))
            outcomes.append(Outcome(probability, _flatten_effect(form[i + 1])))
        total = sum(outcome.probability for outcome in outcomes)
        if total <= 0:
            raise ValueError("Probabilistic action has no positive probability")
        return outcomes

    return [Outcome(1.0, _flatten_effect(effect_expr))]


def parse_domain_text(text: str) -> Domain:
    root = _as_list(parse_sexpr(text), "domain")
    if not root or str(root[0]).lower() != "define":
        raise ValueError("Expected '(define ...)'")

    domain_name = None
    requirements: List[str] = []
    predicates: List[PredicateDecl] = []
    actions: List[ActionSchema] = []

    for entry in root[1:]:
        form = _as_list(entry, "domain entry")
        if not form:
            continue

        head = str(form[0]).lower()
        if head == "domain":
            domain_name = str(form[1])
        elif head == ":requirements":
            requirements = [str(x) for x in form[1:]]
        elif head == ":predicates":
            for pred_expr in form[1:]:
                pred = _as_list(pred_expr, "predicate declaration")
                predicates.append(
                    PredicateDecl(str(pred[0]), _typed_symbols(pred[1:]))
                )
        elif head == ":action":
            name = str(form[1])
            fields = {}
            i = 2
            while i < len(form):
                key = str(form[i]).lower()
                if not key.startswith(":") or i + 1 >= len(form):
                    raise ValueError(f"Malformed action {name} near {form[i:]}")
                fields[key] = form[i + 1]
                i += 2

            parameters = _typed_symbols(
                _as_list(fields.get(":parameters", []), f"{name} parameters")
            )
            preconditions = _flatten_condition(
                fields.get(":precondition", ["and"])
            )
            outcomes = _parse_outcomes(fields.get(":effect", ["and"]))
            actions.append(ActionSchema(name, parameters, preconditions, outcomes))

    if domain_name is None:
        raise ValueError("Domain name not found")

    return Domain(domain_name, requirements, predicates, actions)


def parse_domain(path: str | Path) -> Domain:
    return parse_domain_text(Path(path).read_text())


def parse_problem_text(text: str) -> Problem:
    root = _as_list(parse_sexpr(text), "problem")
    if not root or str(root[0]).lower() != "define":
        raise ValueError("Expected '(define ...)'")

    name = None
    domain_name = None
    objects: List[TypedSymbol] = []
    initial_literals: List[Literal] = []
    goal_conditions: List[Condition] = []

    for entry in root[1:]:
        form = _as_list(entry, "problem entry")
        if not form:
            continue

        head = str(form[0]).lower()
        if head == "problem":
            name = str(form[1])
        elif head == ":domain":
            domain_name = str(form[1])
        elif head == ":objects":
            objects = _typed_symbols(form[1:])
        elif head == ":init":
            for item in form[1:]:
                conditions = _flatten_condition(item)
                for condition in conditions:
                    if not isinstance(condition, Literal):
                        raise ValueError("Equality is not supported in :init")
                    initial_literals.append(condition)
        elif head == ":goal":
            goal_conditions = _flatten_condition(form[1])

    if name is None or domain_name is None:
        raise ValueError("Problem name or domain name missing")

    return Problem(name, domain_name, objects, initial_literals, goal_conditions)


def parse_problem(path: str | Path) -> Problem:
    return parse_problem_text(Path(path).read_text())


def literal_to_pddl(literal: Literal) -> str:
    atom = "(" + " ".join((literal.predicate, *literal.arguments)) + ")"
    return atom if literal.positive else f"(not {atom})"


def condition_to_pddl(condition: Condition) -> str:
    if isinstance(condition, Literal):
        return literal_to_pddl(condition)
    inner = f"(= {condition.left} {condition.right})"
    return inner if condition.equal else f"(not {inner})"