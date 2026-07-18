(define (domain controlled-choice)
  (:requirements :strips :typing :negative-preconditions :probabilistic-effects)

  (:predicates
    (start ?x - object)
    (goal ?x - object)
    (dead ?x - object)
  )

  (:action low_success
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (probabilistic
      0.20 (and
        (goal ?x)
        (not (start ?x))
      )
      0.80 (and
        (dead ?x)
        (not (start ?x))
      )
    )
  )

  (:action high_success
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (probabilistic
      0.80 (and
        (goal ?x)
        (not (start ?x))
      )
      0.20 (and
        (dead ?x)
        (not (start ?x))
      )
    )
  )
)
