         PROGRAM main
         implicit none
         INCLUDE 'mpif.h'
         integer n1,n2
         PARAMETER (n1 = 1, n2 = 1000)
         REAL, ALLOCATABLE :: a(:)
         real ssum,sum
         integer i, ista,iend,myrank,nprocs,ierr

         CALL MPI_INIT(ierr)
         CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
         CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
         CALL para_range(n1, n2, nprocs, myrank, ista, iend)
         ALLOCATE (a(ista:iend))
         DO i = ista, iend
         a(i) = i
         ENDDO
         sum = 0.0
         DO i = ista, iend
         sum = sum + a(i)
        ENDDO
         DEALLOCATE (a)
        CALL MPI_ALLREDUCE(sum, ssum, 1, MPI_REAL,MPI_SUM, MPI_COMM_WORLD, ierr)
!        sum = ssum
!       if(myrank ==0) PRINT *,'sum =',sum
        PRINT *,'sum =',sum
        CALL MPI_FINALIZE(ierr)
        END


       SUBROUTINE para_range(n1, n2, nprocs, irank, ista, iend)
       implicit none
       integer n1,n2,nprocs,irank,ista,iend
       integer iwork
       iwork = (n2 - n1) / nprocs + 1
       ista = MIN(irank * iwork + n1, n2 + 1)
       iend = MIN(ista + iwork - 1, n2)
       END

