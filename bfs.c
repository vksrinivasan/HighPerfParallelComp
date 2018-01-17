#include <petscsys.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include "bfs.h"

int64_t computeNumVertices(MPI_Comm comm, size_t numEdgesLocal, const int64_t (*eLocal_p)[2]);
void determineOwnership(MPI_Comm comm, int *ownership, int64_t *startVertex, int64_t *nOwnVert, int64_t numVertices);
void fillAdjAndCounts(int64_t **eLocal_p_mine, int64_t *adjacencies,
  int64_t *degreeCount, int *ownership, int *sCounts_adj, int *sDispls_adj,
  int64_t numEdgesLocal);
void fillDegCounts(int *ownership, int *sCounts_deg, int *sDispls_deg, int64_t numVertices);
void resortAdjacencies(int64_t *recv_adj, int64_t *recv_deg, int64_t *adj_idx, int64_t startVertex, int64_t nOwnVert, int64_t totAdjIn, int size, int *rDispls_adj);
int edgeCmpr(const void* a, const void* b);
int tEdgeCmpr(const void* a, const void* b);

void initializeStructures(int64_t *locLoopCond, int64_t ***visited, int64_t ***currentQueue, int64_t **currentQueueIndex,
  int64_t **currentQueueSize, int64_t ***nextQueue, int64_t **nextQueueIndex, int64_t **nextQueueSize,
  int64_t ****sendBuffer, int ***sendBufferIndex, int ***sendBufferSize, int **sendSizes_gr,
  int **recvSizes_gr, int **recvDispls_gr, int **sendSizes, int **recvSizes, int **sendDispls, int **recvDispls,
  int64_t **entriesPerKey, int64_t numKeySimul, int64_t nOwnVert, int64_t numVertices, int size);

void coalesceSendSizes(int *sendSizes, int **sendBufferIndex, int64_t numKeySimul, int size);
void compressSendRecv(int *sendSizes_gr, int *recvSizes_gr, int *sendSizes,
  int *recvSizes, int64_t numKeySimul, int size);
void flattenSendData(int64_t *sendBufferFlat, int64_t ***sendBuffer, int **sendBufferIndex,
  int64_t numKeySimul, int size);
void realignRecvData(int64_t **receiveBufferFlat_rearg, int64_t *receiveBufferFlat,
  int *recvSizes_gr, int *recvDispls_gr, int64_t numKeySimul, int size, int64_t totReceiveSize,
  int64_t *entriesPerKey);
void calculateRecvDispls_gr(int *recvSizes_gr, int *recvDispls_gr, int64_t numKeySimul, int size);
void freeStructures(int64_t ***visited, int64_t ***currentQueue, int64_t **currentQueueIndex,
  int64_t **currentQueueSize, int64_t ***nextQueue, int64_t **nextQueueIndex, int64_t **nextQueueSize,
  int64_t ****sendBuffer, int ***sendBufferIndex, int ***sendBufferSize, int **sendSizes_gr,
  int **recvSizes_gr, int **recvDispls_gr, int **sendSizes, int **recvSizes, int **sendDispls, int **recvDispls,
  int64_t **entriesPerKey, int64_t numKeySimul, int size);


int notInCurr(int64_t *current, int64_t startIndex, int64_t currIndex, int64_t vNeigh);
void resizeSingleQueue(int64_t **queue, int64_t *queueIndex, int64_t *queueSize);
void resizeBuffer(int64_t **toOthers, int *index, int *size);
void resizeQueues(int64_t **current, int64_t *currentSize, int64_t nextIndex);

struct _bfsgraph
{
  MPI_Comm  comm;
  int       numProcs;
  int64_t  *adjacencies;   // Array of vertex adjacent vertices to each vertex
  int64_t  *adjacency_idx; // Start index for each vertex into adjacencies
  int      *ownership;     // Which process owns each vertex
  int64_t   startVertex;   // First vertex the proc owns
  int64_t   nOwnVert;      // Number of vertices proc owns
  int64_t   numVertices;   // Number of total vertices
  int64_t   numKeyTog;     // Number of keys to run together at the same time
};

int BFSGraphCreate(MPI_Comm comm, BFSGraph *graph_p)
{
  BFSGraph       graph = NULL;
  PetscErrorCode ierr;
  int            size;

  PetscFunctionBeginUser;

  // Allocate memory for graph, find the number of procs, and then just
  // initialize graph struct
  ierr = PetscCalloc1(1,&graph); CHKERRQ(ierr);

  MPI_Comm_size(comm, &size);

  graph->comm = comm;
  graph->numProcs = size;
  graph->adjacencies = NULL;
  graph->adjacency_idx = NULL;
  graph->ownership = NULL;
  graph->startVertex = -1;
  graph->nOwnVert = -1;
  graph->numVertices = -1;
  graph->numKeyTog = 64;

  *graph_p = graph;
  PetscFunctionReturn(0);
}

int BFSGraphDestroy(BFSGraph *graph_p)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Free malloc'd graph structures - Note that degree gets
  // freed during execution
  free(graph_p[0]->adjacencies);
  free(graph_p[0]->adjacency_idx);
  free(graph_p[0]->ownership);

  ierr = PetscFree(*graph_p); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int BFSGraphGetEdgeArray(BFSGraph graph, size_t E, size_t *numEdgesLocal, int64_t (**elocal_p)[2])
{
  MPI_Comm       comm;
  int            size, rank;
  size_t         edgeStart, edgeEnd;
  size_t         numLocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  comm = graph->comm;

  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

  edgeStart = (rank * E) / size;
  edgeEnd   = ((rank + 1) * E) / size;
  numLocal  = edgeEnd - edgeStart;

  ierr = PetscMalloc1(numLocal, elocal_p); CHKERRQ(ierr);
  *numEdgesLocal = numLocal;

  PetscFunctionReturn(0);
}

int BFSGraphRestoreEdgeArray(BFSGraph graph, size_t E, size_t *numEdgesLocal, int64_t (**elocal_p)[2])
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscFree(*elocal_p); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int BFSGraphSetEdges(BFSGraph graph, size_t E, size_t numEdgesLocal, const int64_t (*eLocal_p)[2])
{
  PetscFunctionBeginUser;
  int size, rank;
  MPI_Comm_size(graph->comm, &size);
  MPI_Comm_rank(graph->comm, &rank);

  // Adjust graph->numKeyTog based on E
  if(E >= 2000000) {
    graph->numKeyTog = 16;
  } else {
    graph->numKeyTog = 64;
  }

  // First make copy of local edges that we can use - We will create new
  // edges for the other direction of the edge (i.e. a-b is given, but we
  // also create b-a)
  int64_t **eLocal_p_mine = (int64_t **)malloc(sizeof(int64_t *)*numEdgesLocal*2);
  for(int64_t i = 0; i < numEdgesLocal*2; i++) {
    eLocal_p_mine[i] = (int64_t *)malloc(sizeof(int64_t)*2);
  }

  for(int64_t i = 0; i < numEdgesLocal*2; i++) {
    if(i < numEdgesLocal) {
      eLocal_p_mine[i][0] = eLocal_p[i][0];
      eLocal_p_mine[i][1] = eLocal_p[i][1];
    } else {
      eLocal_p_mine[i][0] = eLocal_p_mine[i-numEdgesLocal][1];
      eLocal_p_mine[i][1] = eLocal_p_mine[i-numEdgesLocal][0];
    }
  }

  // Get total number of vertices in graph
  int64_t numVertices = computeNumVertices(graph->comm, numEdgesLocal, eLocal_p);
  graph->numVertices = numVertices;

  // Determine ownership for vertices - we will need this for bfs
  int *ownership = (int *)malloc(sizeof(int)*numVertices);
  int64_t startVertex = -1;
  int64_t nOwnVert = 0;
  determineOwnership(graph->comm, ownership, &startVertex, &nOwnVert, numVertices);

  // Sort the local edges array we copied
  qsort(eLocal_p_mine, numEdgesLocal*2, sizeof(eLocal_p_mine[0]), edgeCmpr);

  // Write the adjacencies - Note, while we are writing adjacencies, we can also
  // get data needed for the forthcoming alltoallv's
  int64_t *adjacencies = (int64_t *)malloc(sizeof(int64_t)*numEdgesLocal*2);
  int64_t *degreeCount = (int64_t *)malloc(sizeof(int64_t)*numVertices);

  memset(adjacencies, -1, sizeof(int64_t)*numEdgesLocal*2);
  memset(degreeCount, 0, sizeof(int64_t)*numVertices);

  int     sCounts_adj[size], sDispls_adj[size], rCounts_adj[size], rDispls_adj[size],
          sCounts_deg[size], sDispls_deg[size], rCounts_deg[size], rDispls_deg[size];

  memset(&sCounts_adj, 0, sizeof(int)*size);
  memset(&sDispls_adj, 0, sizeof(int)*size);
  memset(&sCounts_deg, 0, sizeof(int)*size);
  memset(&sDispls_deg, 0, sizeof(int)*size);

  fillAdjAndCounts(eLocal_p_mine, adjacencies, degreeCount, ownership, (int *)&sCounts_adj, (int *)&sDispls_adj, numEdgesLocal*2);
  fillDegCounts(ownership, (int *)&sCounts_deg, (int *)&sDispls_deg, numVertices);

  // Do some alltoalls to figure how how big to make the receive buffer - Also
  // update some receive sizes related things
  MPI_Alltoall(&sCounts_adj, 1, MPI_INT, &rCounts_adj, 1, MPI_INT, graph->comm);
  MPI_Alltoall(&sCounts_deg, 1, MPI_INT, &rCounts_deg, 1, MPI_INT, graph->comm);

  int64_t totAdjIn = 0;
  int64_t totDegIn = 0;

  for(int i = 0; i < size; i++) {
    totAdjIn += rCounts_adj[i];
    totDegIn += rCounts_deg[i];

    if(i == 0) {
      rDispls_adj[i] = 0;
      rDispls_deg[i] = 0;
    } else {
      rDispls_adj[i] = rDispls_adj[i-1] + rCounts_adj[i-1];
      rDispls_deg[i] = rDispls_deg[i-1] + rCounts_deg[i-1];
    }
  }

  int64_t *recv_adj = (int64_t *)malloc(sizeof(int64_t)*totAdjIn);
  int64_t *recv_deg = (int64_t *)malloc(sizeof(int64_t)*totDegIn);

  // Alltoallv
  const int *sCounts_adj_const = (int *)&sCounts_adj;
  const int *sDispls_adj_const = (int *)&sDispls_adj;
  const int *sCounts_deg_const = (int *)&sCounts_deg;
  const int *sDispls_deg_const = (int *)&sDispls_deg;

  const int *rCounts_adj_const = (int *)&rCounts_adj;
  const int *rDispls_adj_const = (int *)&rDispls_adj;
  const int *rCounts_deg_const = (int *)&rCounts_deg;
  const int *rDispls_deg_const = (int *)&rDispls_deg;

  MPI_Alltoallv((const void *)adjacencies, sCounts_adj_const, sDispls_adj_const,
    MPI_INT64_T, recv_adj, rCounts_adj_const, rDispls_adj_const, MPI_INT64_T, graph->comm);

  MPI_Alltoallv((const void *)degreeCount, sCounts_deg_const, sDispls_deg_const,
    MPI_INT64_T, recv_deg, rCounts_deg_const, rDispls_deg_const, MPI_INT64_T, graph->comm);

  free(adjacencies);
  free(degreeCount);
  for(int64_t z = 0; z < numEdgesLocal*2; z++) {
    free(eLocal_p_mine[z]);
  }
  free(eLocal_p_mine);

  // Reorder adjacencies so they are sorted by vertex
  int64_t *adj_idx = (int64_t *)malloc(sizeof(int64_t)*(nOwnVert+1));
  memset(adj_idx, 0, sizeof(int64_t)*(nOwnVert+1));
  resortAdjacencies(recv_adj, recv_deg, adj_idx, startVertex, nOwnVert, totAdjIn, size, (int *)&rDispls_adj);

  // Finally just cumul. sum the adj_indx so we can use it to index into recv_adj
  for(int64_t i = 0; i < (nOwnVert + 1); i++) {
    if(i > 0) {
      adj_idx[i] += adj_idx[i-1];
    }
  }

  // Put variables in the graph for bfs to occur
  graph->adjacencies = recv_adj;
  graph->adjacency_idx = adj_idx;
  graph->ownership = ownership;
  graph->startVertex = startVertex;
  graph->nOwnVert = nOwnVert;

  PetscFunctionReturn(0);
}

int edgeCmpr(const void *pa, const void *pb) {
    const int64_t *a = *(const int64_t **)pa;
    const int64_t *b = *(const int64_t **)pb;
    if(a[0] == b[0])
        return a[1] - b[1];
    else
        return a[0] - b[0];
}

int BFSGraphGetParentArray(BFSGraph graph, size_t *numVerticesLocal, int64_t *firstLocalVertex, int64_t **parentsLocal)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;



  *numVerticesLocal = (size_t)graph->nOwnVert;
  *firstLocalVertex = graph->startVertex;
  parentsLocal[0] = (int64_t *)malloc(sizeof(int64_t)*graph->nOwnVert);
  assert(parentsLocal[0]);

  PetscFunctionReturn(0);
}

int BFSGraphRestoreParentArray(BFSGraph graph, size_t *numVerticesLocal, int64_t *firstLocalVertex, int64_t **parentsLocal)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  free(parentsLocal[0]);

  PetscFunctionReturn(0);
}

int BFSGraphSearch(BFSGraph graph, int num_keys, const int64_t *key, size_t numVerticesLocal, int64_t firstLocalVertex, int64_t **parentsLocal)
{
  PetscFunctionBeginUser;

  int size, rank;
  MPI_Comm_size(graph->comm, &size);
  MPI_Comm_rank(graph->comm, &rank);

  int64_t    numKeySimul = graph->numKeyTog;  // Number of keys to run simultaneously. Must be multiple of 2
  int64_t    numReduceSkip = 5;               // Number frontier expansions that occur before running Allreduce

  int64_t    loopCondition;            //this will be condition for the termination of main while loop
  int64_t    locLoopCond[numKeySimul]; //loop condition for each key running
  int64_t  **visited;                  //keeps track of visited vertices - one for each key being run
  int64_t  **currentQueue;             //vertices that are to be explored in the current iteration - one for each key being run
  int64_t   *currentQueueIndex;        //points to the index in currentQueue where the next vertex is to be inserted - one for each key being run
  int64_t   *currentQueueSize;         //size of currentQueue - one for each key being run
  int64_t  **nextQueue;                //vertices that are to be explored in the next iteration - one for each key being run
  int64_t   *nextQueueIndex;           //points to the index in nextQueue where the next vertex is to be inserted - one for each key being run
  int64_t   *nextQueueSize;            //size of nextQueue - one for each key being run
  int64_t ***sendBuffer;               //2D array that holds vertices that don't belong to current proc and need to be send to corresponding proc - One for each key being run
  int      **sendBufferIndex;          //Array of size,size.points to the index in each sendBuffer array where the next vertex is to be inserted - One for each key being run
  int      **sendBufferSize;           //size of each sendBuffer arrays - One for each key being run
  int       *sendSizes_gr;             // array of sizes we send to each proc
  int       *recvSizes_gr;             // array of sizes we receive from each proc (informs alltoall)
  int       *recvDispls_gr;            // array of recv displacements
  int       *sendSizes;                // array of total sizes we send
  int       *recvSizes;                // array of total sizes we receive
  int       *sendDispls;               // array of send displacements
  int       *recvDispls;               // array of receive displacements
  int64_t   *entriesPerKey;            // How many entries per key were sent to us

  for(int globalKey = 0; globalKey < num_keys; globalKey += (int)numKeySimul) {

    if((globalKey + (int)numKeySimul) >= num_keys) {
      numKeySimul = (int64_t)num_keys - (int64_t)globalKey;
    }

    /* Set parent vertices for all keys we are about to run to -1 */
    for(int keyNum = globalKey; keyNum < (globalKey + (int)numKeySimul); keyNum++) {
      memset(parentsLocal[keyNum], -1, sizeof(int64_t)*graph->nOwnVert);
    }

    /* Initialize all structure we will need */
    initializeStructures((int64_t *)&locLoopCond, &visited, &currentQueue, &currentQueueIndex,
      &currentQueueSize, &nextQueue, &nextQueueIndex, &nextQueueSize, &sendBuffer,
      &sendBufferIndex, &sendBufferSize, &sendSizes_gr, &recvSizes_gr, &recvDispls_gr,
      &sendSizes, &recvSizes, &sendDispls,&recvDispls, &entriesPerKey,
      numKeySimul, graph->nOwnVert, graph->numVertices, size);

    /* Find the procs that have the source vertices for the keys we are currently running
     * Add these to the current queues, and mark their parents as themselves
     */
    for(int keyNum = globalKey; keyNum < (globalKey + (int)numKeySimul); keyNum++) {
      int currKeyIndex = keyNum - globalKey;
      if(graph->ownership[key[keyNum]] == rank) {
        currentQueue[currKeyIndex][currentQueueIndex[currKeyIndex]] = key[keyNum];
        currentQueueIndex[currKeyIndex] += 1;
        parentsLocal[keyNum][key[keyNum] - firstLocalVertex] = key[keyNum];
      }
    }

    /* Set everyone's loop condition to 1 since we know someone has the search
     * keys to start with
     */
    for(int i = 0; i < (int)numKeySimul; i++) locLoopCond[i] = 1;
    loopCondition = 1; // loopCondition is the max of locLoopCond

    /* Now run the sequential while loop. Note, we will do the local work for
     * numKeySimul keys during each while iteration
     */
    int64_t tempCounter = -1;
    while(loopCondition != 0) {
      tempCounter += 1;
      for(int keyNum = globalKey; keyNum < (globalKey + (int)numKeySimul); keyNum++) {
        int currKeyIndex = keyNum - globalKey; // index into which numKeySimul

        /* iterate through all the elements in the relevant current queue */
        for(int64_t j = 0; j < currentQueueIndex[currKeyIndex]; j++) {

          /* "pop" element from the queue and mark it as visited */
          int64_t currVertex = currentQueue[currKeyIndex][j];
          int64_t localVertex = currVertex - firstLocalVertex;
          visited[currKeyIndex][currVertex] = 1;

          int64_t neighStart = graph->adjacency_idx[localVertex];
          int64_t neighEnd = graph->adjacency_idx[localVertex+1];

          /* iterate through allneighbors of the current vertex */
          for(int64_t k = neighStart; k < neighEnd; k++) {
            int64_t neigh = graph->adjacencies[k];
            int64_t localNeigh = neigh - firstLocalVertex;
            int64_t owner = graph->ownership[neigh];

            /* if the current vertex is not the owner of this neighbor, add it
             * to the sendBuffer along with its parent.
             */
            if( (owner != rank) && (visited[currKeyIndex][neigh] == 0) ) {
              resizeBuffer(&sendBuffer[currKeyIndex][owner], &sendBufferIndex[currKeyIndex][owner],
                &sendBufferSize[currKeyIndex][owner]);
              sendBuffer[currKeyIndex][owner][sendBufferIndex[currKeyIndex][owner]] = neigh;
              sendBuffer[currKeyIndex][owner][sendBufferIndex[currKeyIndex][owner] + 1] = currVertex;
              sendBufferIndex[currKeyIndex][owner] += 2;
              visited[currKeyIndex][neigh] = 1; // Mark pseudo-global visited array as seen
            }

            /* if the curr proc is the owner, we can just handle it directly */
            else {
              if( (visited[currKeyIndex][neigh] != 1) &&
                notInCurr(currentQueue[currKeyIndex], j+1, currentQueueIndex[currKeyIndex], neigh) ) {

                  resizeSingleQueue(&nextQueue[currKeyIndex], &nextQueueIndex[currKeyIndex],
                    &nextQueueSize[currKeyIndex]);
                  nextQueue[currKeyIndex][nextQueueIndex[currKeyIndex]] = neigh;
                  nextQueueIndex[currKeyIndex] += 1;
                  parentsLocal[keyNum][localNeigh] = currVertex;
              }
            }
          }
        }
      }

      /* We have completed this iteration's work for every key we are
       * running simulatenously. Each process now needs to know how many
       * adjacencies it's being given for each key.
       *
       * Note these sizes are more granular than what we need to send/receive
       * the data. This granularity, however, is required when we parse
       * what we've received
       */
      coalesceSendSizes(sendSizes_gr, sendBufferIndex, numKeySimul, size);
      MPI_Alltoall(sendSizes_gr, numKeySimul, MPI_INT, recvSizes_gr, numKeySimul, MPI_INT, graph->comm);
      compressSendRecv(sendSizes_gr, recvSizes_gr, sendSizes, recvSizes, numKeySimul, size);

      // We just need the granular receive displacements to rearrange the array
      // we get form the alltoallv later...
      calculateRecvDispls_gr(recvSizes_gr, recvDispls_gr, numKeySimul, size);

      /* Now just compute the total send/recv sizes & displs to do the alltoallv */
      int64_t totSendSize = 0, totReceiveSize = 0;

      for(int k = 0; k < size; k++) {
        totSendSize += sendSizes[k];
        totReceiveSize += recvSizes[k];
        if(k == 0) {
          sendDispls[0] = 0;
          recvDispls[0] = 0;
        } else {
          sendDispls[k] = sendDispls[k-1] + sendSizes[k-1];
          recvDispls[k] = recvDispls[k-1] + recvSizes[k-1];
        }
      }

      /* Flatten the send buffer into a contiguous memory chunk we can send */
      int64_t *sendBufferFlat = (int64_t *)malloc(sizeof(int64_t)*totSendSize);
      int64_t *receiveBufferFlat = (int64_t *)malloc(sizeof(int64_t)*totReceiveSize);
      flattenSendData(sendBufferFlat, sendBuffer, sendBufferIndex, numKeySimul, size);

      /* Now do the alltoallv */
      MPI_Alltoallv((const void *)sendBufferFlat, sendSizes, sendDispls,
        MPI_INT64_T, receiveBufferFlat, recvSizes, recvDispls, MPI_INT64_T, graph->comm);

      /* Rearrange the receive array we were just given for better cache performance */
      free(sendBufferFlat); // Might help minimize out of memory issues if we do it here
      sendBufferFlat = NULL;
      int64_t *receiveBufferFlat_rearg;
      realignRecvData(&receiveBufferFlat_rearg, receiveBufferFlat, recvSizes_gr,
        recvDispls_gr, numKeySimul, size, totReceiveSize, entriesPerKey);

      /* Now process the received vertices */
      int64_t currentEntryIndex = 0;
      for(int64_t k = 0; k < numKeySimul; k++) {
        for(int64_t z = 0; z < entriesPerKey[k]; z+= 2) {
          int64_t vertex = receiveBufferFlat_rearg[currentEntryIndex];
          int64_t vertexParent = receiveBufferFlat_rearg[currentEntryIndex + 1];

          /* set the parent to visited in pseudo-global visited array */
          visited[k][vertexParent] = 1;

          /* if the vertex hasn't been visited yet, we add it's parent to the
           * parentsLocal array and add it to the next queue
           */
          if(parentsLocal[globalKey + (int)k][vertex - firstLocalVertex] == -1) {
            resizeSingleQueue(&(nextQueue[k]), &(nextQueueIndex[k]), &(nextQueueSize[k]));
            nextQueue[k][nextQueueIndex[k]] = vertex;
            nextQueueIndex[k] += 1;
            parentsLocal[globalKey + (int)k][vertex - firstLocalVertex] = vertexParent;
          }
          currentEntryIndex += 2;
        }
      }

      /* Update local loop condition and global loop condition to see if we still
       * need to work on this set of keys
       */
      if((tempCounter % numReduceSkip) == 0) {
        for(int64_t k = 0; k < numKeySimul; k++) {
          locLoopCond[k] = nextQueueIndex[k];
        }
        MPI_Allreduce(MPI_IN_PLACE, &locLoopCond, (int)numKeySimul, MPI_INT64_T,
          MPI_MAX, graph->comm);

        loopCondition = 0;
        for(int64_t k = 0; k < numKeySimul; k++) {
          if(locLoopCond[k] > 0) {
            loopCondition = 1;
            break;
          }
        }
      } else {
        loopCondition = 1;
      }

      /* Copy the next queues into the current queues and update the index
       * pointers
       */
       for(int64_t k = 0; k < numKeySimul; k++) {
         resizeQueues(&(currentQueue[k]), &(currentQueueSize[k]), nextQueueIndex[k]);
         memcpy(currentQueue[k], nextQueue[k], sizeof(int64_t)*nextQueueIndex[k]);
         currentQueueIndex[k] = nextQueueIndex[k];
         nextQueueIndex[k] = 0;

         /* Just reset the send buffer indices */
         for(int m = 0; m < size; m++) {
           sendBufferIndex[k][m] = 0;
         }
       }

      /* Free the remaining flat array */
      free(receiveBufferFlat_rearg);

      /* clean the send/recv structures that we will reuse */
      memset(sendSizes_gr, 0, sizeof(int)*numKeySimul*((int64_t)size));
      memset(recvSizes_gr, 0, sizeof(int)*numKeySimul*((int64_t)size));
      memset(recvDispls_gr, 0, sizeof(int)*numKeySimul*((int64_t)size));
      memset(sendSizes, 0, sizeof(int)*size);
      memset(recvSizes, 0, sizeof(int)*size);
      memset(sendDispls, 0, sizeof(int)*size);
      memset(recvDispls, 0, sizeof(int)*size);
      memset(entriesPerKey, 0, sizeof(int64_t)*numKeySimul);
    }

    /* Now just free the data structures we used for this set of keys */
    freeStructures(&visited, &currentQueue, &currentQueueIndex,
      &currentQueueSize, &nextQueue, &nextQueueIndex, &nextQueueSize, &sendBuffer,
      &sendBufferIndex, &sendBufferSize, &sendSizes_gr, &recvSizes_gr, &recvDispls_gr,
      &sendSizes, &recvSizes, &sendDispls,&recvDispls, &entriesPerKey,
      numKeySimul, size);
  }
  PetscFunctionReturn(0);
}

int64_t computeNumVertices(MPI_Comm comm, size_t numEdgesLocal, const int64_t (*eLocal_p)[2]) {
  int64_t numVertices = -1;
  for(size_t i = 0; i < numEdgesLocal; i++) {
    if(eLocal_p[i][0] > numVertices)  numVertices = eLocal_p[i][0];
    if(eLocal_p[i][1] > numVertices)  numVertices = eLocal_p[i][1];
  }

  MPI_Allreduce(MPI_IN_PLACE , &numVertices, 1, MPI_INT64_T, MPI_MAX, comm);
  numVertices += 1; // |V| vertices
  return numVertices;
}

// Determine who owns which vertices and create ownership array for reference
void determineOwnership(MPI_Comm comm, int *ownership, int64_t *startVertex, int64_t *nOwnVert, int64_t numVertices) {
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  int64_t vPerProc = numVertices/size;

  if(rank != (size-1)) *nOwnVert = vPerProc;

  int64_t currVertex = 0;
  for(int i = 0; i < size; i++) {
    if(i == rank) *startVertex = currVertex;
    if(i != size-1) {
      for(int j = 0; j < vPerProc; j++) {
        ownership[currVertex] = i;
        currVertex += 1;
      }
    } else {
      if(i == rank) {
        *nOwnVert = numVertices - currVertex;
      }
      for(int64_t j = currVertex; j < numVertices; j++) ownership[j] = i;
    }
  }
}

void fillAdjAndCounts(int64_t **eLocal_p_mine, int64_t *adjacencies,
  int64_t *degreeCount, int *ownership, int *sCounts_adj, int *sDispls_adj,
  int64_t numEdgesLocal) {

  int prevOwner;
  for(int64_t i = 0; i < numEdgesLocal; i++) {
    adjacencies[i] = eLocal_p_mine[i][1]; // Keep track of destinations
    degreeCount[eLocal_p_mine[i][0]] += 1; // Keep track of degree counts (i.e. the source)
    int ownerProc = ownership[eLocal_p_mine[i][0]];
    sCounts_adj[ownerProc] += 1;

    if(i == 0) {
      sDispls_adj[ownerProc] = 0;
      prevOwner = ownerProc;
    } else if(ownerProc != prevOwner) {
      sDispls_adj[ownerProc] = i;
      prevOwner = ownerProc;
    }
  }

}

void fillDegCounts(int *ownership, int *sCounts_deg, int *sDispls_deg, int64_t numVertices) {

  int prevOwner;
  for(int64_t i = 0; i < numVertices; i++) {
    int ownerProc = ownership[i];
    sCounts_deg[ownerProc] += 1;

    if(i == 0) {
      sDispls_deg[ownerProc] = 0;
      prevOwner = ownerProc;
    } else if(ownerProc != prevOwner) {
      sDispls_deg[ownerProc] = i;
      prevOwner = ownerProc;
    }
  }
}

void resortAdjacencies(int64_t *recv_adj, int64_t *recv_deg, int64_t *adj_idx, int64_t startVertex, int64_t nOwnVert,
  int64_t totAdjIn, int size, int *rDispls_adj) {

  int64_t *sortedAdj = (int64_t *)malloc(sizeof(int64_t)*totAdjIn);
  memset(sortedAdj, 0, sizeof(int64_t)*totAdjIn);

  int64_t currSortedIndex = 0;

  for(int64_t i = 0; i < nOwnVert; i++) {
    for(int64_t j = 0; j < size; j++) {
      int64_t degIndex = j*nOwnVert + i;
      int64_t deg = recv_deg[degIndex];

      adj_idx[i+1] += deg;

      if(deg != 0) {
        memcpy(&sortedAdj[currSortedIndex], &recv_adj[rDispls_adj[j]], sizeof(int64_t)*deg);
        currSortedIndex += deg;
        rDispls_adj[j] += deg;
      }
    }
  }

  memcpy(recv_adj, sortedAdj, sizeof(int64_t)*totAdjIn);
  free(sortedAdj);
  free(recv_deg);
  sortedAdj = NULL;
  recv_deg = NULL;
}

int notInCurr(int64_t *current, int64_t startIndex, int64_t currIndex, int64_t vNeigh) {
  for(int64_t i = startIndex; i < currIndex; i++) {
    if(current[i] == vNeigh) return 0;
  }
  return 1;
}

void resizeSingleQueue(int64_t **queue, int64_t *queueIndex, int64_t *queueSize){
  if(*queueIndex == *queueSize) {
    int64_t *newQueue = (int64_t *)malloc(sizeof(int64_t)*(*queueSize)*2);
    memcpy(newQueue, (*queue), sizeof(int64_t)*(*queueSize));
    memset(&(newQueue[*queueIndex]), -1, sizeof(int64_t)*(*queueSize));
    *queueSize = (*queueSize)*2;
    int64_t *temp = (*queue);
    (*queue) = newQueue;
    free(temp);
    temp = NULL;
  }
}


void resizeBuffer(int64_t **toOthers, int *index, int *size){
  if((*index + 2) >= *size) {
    int64_t *newToOthers = (int64_t *)malloc(sizeof(int64_t)*(*size)*2);
    memcpy(newToOthers, (*toOthers), sizeof(int64_t)*(*size));
    int numToClean = (*size * 2) - *index;
    memset(&(newToOthers[*index]), -1, sizeof(int64_t)*(numToClean));
    *size = (*size)*2;
    int64_t *temp = (*toOthers);
    (*toOthers) = newToOthers;
    free(temp);
    temp = NULL;
  }
}


void resizeQueues(int64_t **current, int64_t *currentSize, int64_t nextIndex){

  if(nextIndex > *currentSize) {
    int64_t *newCurrent = (int64_t *)malloc(sizeof(int64_t)*nextIndex);
    int64_t *temp = (*current);
    (*current) = newCurrent;
    *currentSize = nextIndex;
    free(temp);
    temp = NULL;
  }
}

void initializeStructures(int64_t *locLoopCond, int64_t ***visited, int64_t ***currentQueue, int64_t **currentQueueIndex,
  int64_t **currentQueueSize, int64_t ***nextQueue, int64_t **nextQueueIndex, int64_t **nextQueueSize,
  int64_t ****sendBuffer, int ***sendBufferIndex, int ***sendBufferSize, int **sendSizes_gr,
  int **recvSizes_gr, int **recvDispls_gr, int **sendSizes, int **recvSizes, int **sendDispls, int **recvDispls,
  int64_t **entriesPerKey, int64_t numKeySimul, int64_t nOwnVert, int64_t numVertices, int size) {

    /* Initialize the local loop conditions to 0 */
    for(int64_t i = 0; i < numKeySimul; i++) {
      locLoopCond[i] = 0;
    }

    /* Allocate space for the current arrays */
    *visited = (int64_t **)malloc(sizeof(int64_t *)*numKeySimul);
    assert(*visited);
    for(int64_t i = 0; i < numKeySimul; i++) {
      (*visited)[i] = (int64_t *)malloc(sizeof(int64_t)*(numVertices));
      memset((*visited)[i], 0, sizeof(int64_t)*numVertices);
      assert(((*visited)[i]));
    }

    /* Allocate space for the current queues */
    *currentQueue = (int64_t**)malloc(sizeof(int64_t *)*numKeySimul);
    *currentQueueIndex = (int64_t *)malloc(sizeof(int64_t)*numKeySimul);
    *currentQueueSize = (int64_t *)malloc(sizeof(int64_t)*numKeySimul);

    assert(*currentQueue);
    assert(*currentQueueIndex);
    assert(*currentQueueSize);

    for(int64_t i = 0; i < numKeySimul; i++) {
      (*currentQueue)[i] = (int64_t *)malloc(sizeof(int64_t)*(nOwnVert));
      assert(((*currentQueue)[i]));
      memset((*currentQueue)[i], -1, sizeof(int64_t)*(nOwnVert));
      (*currentQueueIndex)[i] = 0;
      (*currentQueueSize)[i] = nOwnVert;
    }

    /* Allocate space for the next queue */
    *nextQueue = (int64_t **)malloc(sizeof(int64_t *)*(numKeySimul));
    *nextQueueIndex = (int64_t *)malloc(sizeof(int64_t)*numKeySimul);
    *nextQueueSize = (int64_t *)malloc(sizeof(int64_t)*numKeySimul);

    assert(*nextQueue);
    assert(*nextQueueIndex);
    assert(*nextQueueSize);

    for(int64_t i = 0; i < numKeySimul; i++) {
      (*nextQueue)[i] = (int64_t *)malloc(sizeof(int64_t)*(nOwnVert));
      assert(((*nextQueue)[i]));
      memset((*nextQueue)[i], -1, sizeof(int64_t)*(nOwnVert));
      (*nextQueueIndex)[i] = 0;
      (*nextQueueSize)[i] = nOwnVert;
    }

    /* Allocate space for the send buffers */
    *sendBuffer = (int64_t ***)malloc(sizeof(int64_t **)*numKeySimul);
    *sendBufferIndex = (int **)malloc(sizeof(int *)*numKeySimul);
    *sendBufferSize = (int **)malloc(sizeof(int *)*numKeySimul);

    assert(*sendBuffer);
    assert(*sendBufferIndex);
    assert(*sendBufferSize);

    for(int64_t i = 0; i < numKeySimul; i++) {
      (*sendBuffer)[i] = (int64_t **)malloc(sizeof(int64_t *)*size);
      (*sendBufferIndex)[i] = (int *)malloc(sizeof(int)*size);
      (*sendBufferSize)[i] = (int *)malloc(sizeof(int)*size);

      assert((*sendBuffer)[i]);
      assert((*sendBufferIndex)[i]);
      assert((*sendBufferSize)[i]);

      for(int j = 0; j < size; j++) {
        (*sendBuffer)[i][j] = (int64_t *)malloc(sizeof(int64_t)*(nOwnVert));
        assert((*sendBuffer)[i][j]);
        memset((*sendBuffer)[i][j], -1, sizeof(int64_t)*nOwnVert);
        (*sendBufferIndex)[i][j] = 0;
        (*sendBufferSize)[i][j] = nOwnVert;
      }
    }

    /* Allocate space for send/recv sizes and displs */
    *sendSizes_gr = (int *)malloc(sizeof(int)*numKeySimul*((int64_t)size));
    *recvSizes_gr = (int *)malloc(sizeof(int)*numKeySimul*((int64_t)size));
    *recvDispls_gr = (int *)malloc(sizeof(int)*numKeySimul*((int64_t)size));
    *sendSizes = (int *)malloc(sizeof(int)*size);
    *recvSizes = (int *)malloc(sizeof(int)*size);
    *sendDispls = (int *)malloc(sizeof(int)*size);
    *recvDispls = (int *)malloc(sizeof(int)*size);

    assert(*sendSizes_gr);
    assert(*recvSizes_gr);
    assert(*recvDispls_gr);
    assert(*sendSizes);
    assert(*recvSizes);
    assert(*sendDispls);
    assert(*recvDispls);

    for(int64_t i = 0; i < numKeySimul*((int64_t)size); i++) {
      (*sendSizes_gr)[i] = 0;
      (*recvSizes_gr)[i] = 0;
      (*recvDispls_gr)[i] = 0;
    }
    for(int i = 0; i < size; i++) {
      (*sendSizes)[i] = 0;
      (*recvSizes)[i] = 0;
      (*sendDispls)[i] = 0;
      (*recvDispls)[i] = 0;
    }

    /* Allocate space for entries per key */
    *entriesPerKey = (int64_t *)malloc(sizeof(int64_t)*numKeySimul);
    memset(*entriesPerKey, 0, sizeof(int64_t)*numKeySimul);

}

void coalesceSendSizes(int *sendSizes, int **sendBufferIndex, int64_t numKeySimul, int size) {
  int64_t currIndex = 0;
  for(int i = 0; i < size; i++) {
    for(int64_t j = 0; j < numKeySimul; j++) {
      sendSizes[currIndex] = sendBufferIndex[j][i];
      currIndex += 1;
    }
  }
}

void compressSendRecv(int *sendSizes_gr, int *recvSizes_gr, int *sendSizes,
  int *recvSizes, int64_t numKeySimul, int size) {

    for(int64_t i = 0; i < numKeySimul*((int64_t)size); i += numKeySimul) {
      for(int64_t j = 0; j < numKeySimul; j++) {
        int64_t index = (i/numKeySimul);
        sendSizes[index] += sendSizes_gr[i+j];
        recvSizes[index] += recvSizes_gr[i+j];
      }
    }

}

void flattenSendData(int64_t *sendBufferFlat, int64_t ***sendBuffer, int **sendBufferIndex,
  int64_t numKeySimul, int size) {

    int64_t index = 0;
    for(int i = 0; i < size; i++) {
      for(int64_t j = 0; j < numKeySimul; j++) {
        memcpy(&sendBufferFlat[index], sendBuffer[j][i], sizeof(int64_t)*sendBufferIndex[j][i]);
        index += sendBufferIndex[j][i];
      }
    }
}

void realignRecvData(int64_t **receiveBufferFlat_rearg, int64_t *receiveBufferFlat,
  int *recvSizes_gr, int *recvDispls_gr, int64_t numKeySimul, int size, int64_t totReceiveSize,
  int64_t *entriesPerKey) {

    *receiveBufferFlat_rearg = (int64_t *)malloc(sizeof(int64_t)*totReceiveSize);
    int64_t index = 0;
    for(int64_t i = 0; i < numKeySimul; i++) {
      for(int64_t j = i; j < (numKeySimul * ((int64_t)size)); j+=numKeySimul) {
        int rBufFlatIdx = recvDispls_gr[j];
        memcpy(&(*receiveBufferFlat_rearg)[index], &receiveBufferFlat[rBufFlatIdx],
          sizeof(int64_t)*recvSizes_gr[j]);

        index += recvSizes_gr[j];
        entriesPerKey[i] += recvSizes_gr[j];
      }
    }
    free(receiveBufferFlat);
    receiveBufferFlat = NULL;
}

void calculateRecvDispls_gr(int *recvSizes_gr, int *recvDispls_gr, int64_t numKeySimul, int size) {
  for(int64_t k = 0; k < numKeySimul*((int64_t)size); k++) {
    if(k == 0) {
      recvDispls_gr[0] = 0;
    } else {
      recvDispls_gr[k] = recvDispls_gr[k-1] + recvSizes_gr[k-1];
    }
  }
}

void freeStructures(int64_t ***visited, int64_t ***currentQueue, int64_t **currentQueueIndex,
  int64_t **currentQueueSize, int64_t ***nextQueue, int64_t **nextQueueIndex, int64_t **nextQueueSize,
  int64_t ****sendBuffer, int ***sendBufferIndex, int ***sendBufferSize, int **sendSizes_gr,
  int **recvSizes_gr, int **recvDispls_gr, int **sendSizes, int **recvSizes, int **sendDispls, int **recvDispls,
  int64_t **entriesPerKey, int64_t numKeySimul, int size) {

    for(int64_t m = 0; m < numKeySimul; m++) {
      for(int z = 0; z < size; z++) {
        free((*sendBuffer)[m][z]);
        (*sendBuffer)[m][z] = NULL;
      }
      free((*visited)[m]);
      (*visited)[m] = NULL;
      free((*currentQueue)[m]);
      (*currentQueue)[m] = NULL;
      free((*nextQueue)[m]);
      (*nextQueue)[m] = NULL;
      free((*sendBuffer)[m]);
      (*sendBuffer)[m] = NULL;
      free((*sendBufferIndex)[m]);
      (*sendBufferIndex)[m] = NULL;
      free((*sendBufferSize)[m]);
      (*sendBufferSize)[m] = NULL;
    }
    free(*visited);
    *visited=NULL;
    free(*sendBuffer);
    *sendBuffer = NULL;
    free(*sendBufferIndex);
    sendBufferIndex = NULL;
    free(*sendBufferSize);
    sendBufferSize = NULL;
    free(*currentQueue);
    *currentQueue = NULL;
    free(*nextQueue);
    *nextQueue = NULL;
    free(*currentQueueIndex);
    *currentQueueIndex = NULL;
    free(*currentQueueSize);
    *currentQueueSize = NULL;
    free(*nextQueueIndex);
    *nextQueueIndex = NULL;
    free(*nextQueueSize);
    *nextQueueSize = NULL;
    free(*sendSizes_gr);
    *sendSizes_gr = NULL;
    free(*recvSizes_gr);
    *recvSizes_gr = NULL;
    free(*recvDispls_gr);
    *recvDispls_gr = NULL;
    free(*sendSizes);
    *sendSizes = NULL;
    free(*recvSizes);
    *recvSizes = NULL;
    free(*sendDispls);
    *sendDispls = NULL;
    free(*recvDispls);
    *recvDispls = NULL;
    free(*entriesPerKey);
    *entriesPerKey = NULL;
}

/* Debugging Functions */
void printEdges(int64_t numEdgesLocal, int64_t **eLocal_p_mine) {
  for(int64_t i = 0; i < numEdgesLocal; i++) {
    printf("%" PRId64 "\t %" PRId64 "\n", eLocal_p_mine[i][0], eLocal_p_mine[i][1]);
  }
}

void printHisEdges(int64_t numEdgesLocal, const int64_t (*eLocal_p)[2]) {
  for(int64_t i = 0; i < numEdgesLocal; i++) {
    printf("%" PRId64 "\t %" PRId64 "\n", eLocal_p[i][0], eLocal_p[i][1]);
  }
}

void printOwnership(int64_t numVertices, int *ownership) {
  for(int64_t i = 0; i < numVertices; i++) {
    printf("vertex: %d owner: %d\n", (int)i, ownership[i]);
  }
}

void printAdjacencies(size_t numEdgesLocal, int64_t *adjacencies) {
  for(size_t i = 0; i < numEdgesLocal; i++) {
    printf("%d\t %"PRId64"\n", (int)i, adjacencies[i]);
  }
}

void printDegreeCounts(int64_t numVertices, int64_t *degrees) {
  for(int64_t i = 0; i < numVertices; i++) {
    printf("Vertex %d\t %"PRId64"\n", (int)i, degrees[i]);
  }
}

void printParents(int64_t *parents, int64_t numVertices, int64_t startVertex) {
  for(int64_t i = 0; i < numVertices; i++) {
    printf("Vertex %"PRId64"\t Parent %"PRId64"\n", startVertex + i, parents[i]);
  }
}
