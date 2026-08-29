abstract type AbstractSourceElement end
abstract type AbstractCompactSourceElement <: AbstractSourceElement end
abstract type AbstractNonCompactSourceElement <: AbstractSourceElement end

"""
    velocity(se::AbstractSourceElement)

Return the current velocity of `se`.
"""
@inline velocity(se::AbstractSourceElement) = se.y1dot

"""
    source_time(se::AbstractSourceElement)

Return the source time of `se`.
"""
@inline source_time(se::AbstractSourceElement) = se.τ

"""
    orientation(se::AbstractCompactSourceElement)

Return a length-3 unit vector indicating the spanwise orientation of `se`.
"""
@inline orientation(se::AbstractCompactSourceElement) = se.span_uvec

"""
    position(se::AbstractSourceElement)

Return a length-3 vector indicating the position of `se`.
"""
@inline position(se::AbstractSourceElement) = se.y0dot

"""
    speed_of_sound(se::AbstractSourceElement)

Return the ambient speed of sound associated with `se`.
"""
@inline speed_of_sound(se) = se.c0
