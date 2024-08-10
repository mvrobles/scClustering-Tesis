package src;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class KMSTWeightedGraphGenerator implements WeightedGraphGenerator {

	@Override
	public List<WeightedEdge> createWeightedGraph(double[][] samplesMatrix) {
		List<DistanceEdge> internalGraph = buildKMSTGraph(samplesMatrix);
		List<WeightedEdge> weightedGraph = new ArrayList<>();
		for(DistanceEdge edge:internalGraph) {
			weightedGraph.add(new WeightedEdge(edge.getV1(), edge.getV2(), (int)(1000*(edge.getCorrelation()))));
		}
		return weightedGraph;
	}
	
	private List<DistanceEdge> buildKMSTGraph(double [][] correlations) {
		List<DistanceEdge> answer = new ArrayList<DistanceEdge>();
		List<DistanceEdge> remainingEdges = new ArrayList<DistanceEdge>();
		int n = correlations.length;
		for(int i=0;i<n;i++) {
			for(int j=i+1;j<n;j++) {
				double d = correlations[i][j];
				remainingEdges.add(new DistanceEdge(i,j,d));
			}
		}
		int rounds = (int) Math.round(Math.log(n));
		System.out.println("Building k-mst graph with k="+rounds);
		for(int k=0;k<rounds;k++) {
			List<DistanceEdge> mst = calculateMST(n,remainingEdges);
			answer.addAll(mst);
			remainingEdges = removeEdges(remainingEdges,mst);
			//remainingEdges.removeAll(mst);
		}
		return answer;
	}
	private List<DistanceEdge> removeEdges(List<DistanceEdge> remainingEdges, List<DistanceEdge> mst) {
		Set<String> ids = new HashSet<String>();
		for(DistanceEdge edge:mst) ids.add(""+edge.getV1()+" "+edge.getV2());
		List<DistanceEdge> answer = new ArrayList<DistanceEdge>();
		for(DistanceEdge edge:remainingEdges) {
			if(!ids.contains(""+edge.getV1()+" "+edge.getV2())) answer.add(edge);
		}
		return answer;
	}

	private List<DistanceEdge> calculateMST(int n, List<DistanceEdge> remainingEdges) {
		List<DistanceEdge> answer = new ArrayList<DistanceEdge>();
		Collections.sort(remainingEdges,(e1,e2)-> e1.getCost()-e2.getCost());
		DisjointSets ds = new DisjointSets(n);
		for(DistanceEdge edge:remainingEdges) {
			if(!ds.sameSubsets(edge.getV1(), edge.getV2())) {
				answer.add(edge);
				ds.union(edge.getV1(), edge.getV2());
				if (ds.getNumSubsets()==1) break;
			}
		}
		return answer;
	}

}
class DistanceEdge {
	private int v1;
	private int v2;
	private double correlation;
	private int cost;
	public DistanceEdge(int v1, int v2, double correlation) {
		super();
		this.v1 = v1;
		this.v2 = v2;
		this.correlation = correlation;
		cost = (int) (10000*(1-correlation));
	}
	public int getV1() {
		return v1;
	}
	public int getV2() {
		return v2;
	}
	public double getCorrelation() {
		return correlation;
	}
	public int getCost() {
		return cost;
	}
}
