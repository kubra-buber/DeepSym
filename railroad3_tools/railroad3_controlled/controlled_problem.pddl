(define (problem controlled-blocks)
    (:domain blocks)
    (:objects obj0 - controlled_role0 obj1 - controlled_role1)
    (:init
        (not_r0 obj0 obj0)
        (not_r0 obj1 obj1)
        (not_r2 obj0 obj0)
        (not_r2 obj1 obj1)
        (r1 obj0 obj0)
        (r1 obj1 obj1)
        (z0 obj0)
        (z0 obj1)
    )
    (:goal
        (and
            (r1 obj0 obj1)
        )
    )
)
