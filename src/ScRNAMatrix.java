package src;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class ScRNAMatrix {
	private List<String> cellIds;
	private List<Map<Integer,Short>> countsByCell= new ArrayList<Map<Integer,Short>>();
	private List<String> geneIds;
	private List<Map<Integer,Short>> countsByGene = new ArrayList<Map<Integer,Short>>();
	
	public ScRNAMatrix(List<String> cellIds, List<String> geneIds, List<List<Integer>> counts) {
		super();
		this.cellIds = cellIds;
		this.geneIds = geneIds;
		initializeCounts(cellIds.size(), geneIds.size());
		for(List<Integer> countList:counts) {
			int cellId = countList.get(0);
			int geneId = countList.get(1);
			short count = (short) Math.min(Short.MAX_VALUE, countList.get(2));
			countsByCell.get(cellId).put(geneId, count);
			countsByGene.get(geneId).put(cellId, count);
		}
	}
	public ScRNAMatrix (int [][] fullMatrix) {
		initializeCounts(fullMatrix.length, fullMatrix[0].length);
		for(int i=0;i<fullMatrix.length;i++) {
			Map<Integer,Short> countsCell = countsByCell.get(i);
			for(int j=0;j<fullMatrix[i].length;j++) {
				Map<Integer,Short> countsGene = countsByGene.get(j);
				int value = fullMatrix[i][j];
				if(value==0) continue;
				if(value >Short.MAX_VALUE) value = Short.MAX_VALUE;
				countsCell.put(j,(short) value);
				countsGene.put(i,(short) value);
			}
		}
	}
	public List<String> getCellIds() {
		return cellIds;
	}
	public List<String> getGeneIds() {
		return geneIds;
	}
	public Map<Integer,Short> getCountsCell(int cellIdx, int minValue) {
		Map<Integer,Short> answer = new TreeMap<Integer, Short>();
		for(Map.Entry<Integer,Short> entry:countsByCell.get(cellIdx).entrySet()) {
			if(entry.getValue()>=minValue) answer.put(entry.getKey(), entry.getValue());
		}
		return answer;
	}
	private void initializeCounts(int numGenes, int numCells) {
		for(int i=0;i<cellIds.size();i++) countsByCell.add(new TreeMap<Integer, Short>());
		for(int i=0;i<geneIds.size();i++) countsByGene.add(new TreeMap<Integer, Short>());	
	}
	
	public void filterGenes() {
		Set<Integer> idxsToRemove = new HashSet<Integer>();
		for(int j=0;j<countsByGene.size();j++) {
			Map<Integer,Short> countsGene= countsByGene.get(j);
			if(countsGene.size()<10) {
				//System.out.println("Removing gene: "+j+" counts: "+countsGene.size());
				idxsToRemove.add(j);
			}
		}
		removeGenes(idxsToRemove);
		System.out.println("Remaining genes: "+countsByGene.size()+" removed: "+idxsToRemove.size());
		
	}
	public void removeGenes(Set<Integer> idxsToRemove) {
		List<String> geneIds2 = new ArrayList<String>();
		List<Map<Integer,Short>> countsByGene2 = new ArrayList<Map<Integer,Short>>();
		int j2 = 0;
		for(int j=0;j<countsByGene.size();j++) {
			boolean b = idxsToRemove.contains(j);
			if(b || j!=j2) {
				//System.out.println("Gene: "+j+" to remove: "+b+" new index: "+j2);
				Map<Integer,Short> countsGene= countsByGene.get(j);
				for(int i:countsGene.keySet()) {
					Map<Integer,Short> countsCell = countsByCell.get(i);
					short count = countsCell.remove(j);
					if(!b) countsCell.put(j2,count);
				}
			}
			if(!b) {
				geneIds2.add(geneIds.get(j));
				countsByGene2.add(countsByGene.get(j));
				j2++;
			}
		}
		geneIds = geneIds2;
		countsByGene = countsByGene2;
	}
	public void filterCells() {
		Set<Integer> idxsToRemove = new HashSet<Integer>();
		for(int i=0;i<countsByCell.size();i++) {
			Map<Integer,Short> countsCell = countsByCell.get(i);
			if(countsCell.size()<50) {
				System.out.println("Removing cell: "+i +" count: "+countsCell.size());
				idxsToRemove.add(i);
			}
		}
		removeCells(idxsToRemove);
		System.out.println("Remaining cells: "+countsByCell.size()+" removed: "+idxsToRemove.size());
	}
	public void removeCells(Set<Integer> idxsToRemove) {
		List<String> cellIds2 = new ArrayList<String>();
		List<Map<Integer,Short>> countsByCell2 = new ArrayList<Map<Integer,Short>>();
		int i2 = 0;
		for(int i=0;i<countsByCell.size();i++) {
			boolean b = idxsToRemove.contains(i); 
			if(b || i!=i2) {
				Map<Integer,Short> countsCell = countsByCell.get(i);
				for(int j:countsCell.keySet()) {
					Map<Integer,Short> countsGene = countsByGene.get(j);
					short count = countsGene.remove(i);
					if(!b) countsGene.put(i2,count);
				}
			}
			if(!b) {
				cellIds2.add(cellIds.get(i));
				countsByCell2.add(countsByCell.get(i));
				i2++;
			}
		}
		cellIds = cellIds2;
		countsByCell = countsByCell2;
	}
}
