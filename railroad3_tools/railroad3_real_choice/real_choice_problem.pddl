(define (problem railroad-real-domain-choice)
  (:domain blocks)

  (:objects
    obj0 obj1
  )

  (:init
    (not_r1 obj0 obj0)
    (not_r1 obj0 obj1)
    (not_r1 obj1 obj0)
    (not_r1 obj1 obj1)
    (r0 obj0 obj0)
    (r0 obj1 obj1)
    (r2 obj0 obj0)
    (r2 obj0 obj1)
    (r2 obj1 obj0)
    (r2 obj1 obj1)
    (z0 obj0)
    (z0 obj1)
  )

  (:goal
    (r1 obj1 obj1)
  )
)
