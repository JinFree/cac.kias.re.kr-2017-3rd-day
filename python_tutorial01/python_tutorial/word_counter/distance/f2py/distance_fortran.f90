subroutine calc_distance(c1_size,n1,c2_size,n2,c1,c2,distance)
   implicit none
   integer, intent(in) :: c1_size, n1, c2_size, n2
   real*8, intent(in), dimension(c1_size,n1) :: c1
   real*8, intent(in), dimension(c2_size,n2) :: c2
   real*8, intent(out), dimension(c1_size,c2_size) :: distance
   
   integer :: i, j
   do i = 1,c1_size
      do j = 1,c2_size
         distance(i,j) = dsqrt( (c1(i,1)-c2(j,1))**2 + (c1(i,2)-c2(j,2))**2 + (c1(i,3)-c2(j,3))**2 )
      end do
   end do
end subroutine
