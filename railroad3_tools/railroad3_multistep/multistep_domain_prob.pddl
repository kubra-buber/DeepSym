(define (domain controlled-multistep)
  (:requirements :strips :typing :probabilistic-effects)

  (:predicates
    (start ?x - object)
    (ready ?x - object)
    (goal ?x - object)
    (dead ?x - object)
  )

  (:action risky_direct
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (probabilistic
      0.55 (and
        (goal ?x)
        (not (start ?x))
      )
      0.45 (and
        (dead ?x)
        (not (start ?x))
      )
    )
  )

  (:action prepare_safe
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (and
      (ready ?x)
      (not (start ?x))
    )
  )

  (:action finish_safe
    :parameters (?x - object)
    :precondition (ready ?x)
    :effect (probabilistic
      0.90 (and
        (goal ?x)
        (not (ready ?x))
      )
      0.10 (and
        (dead ?x)
        (not (ready ?x))
      )
    )
  )
)
