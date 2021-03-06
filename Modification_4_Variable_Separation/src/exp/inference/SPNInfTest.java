package exp.inference;

import spn.GraphSPN;
import util.Parameter;
import data.Dataset;
import data.Partition;
import exp.RunSLSPN;

public class SPNInfTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		RunSLSPN.parseParameters(args);
		String prefix = "data/";
		
		Dataset d = null;
		try {
			d = (Dataset) RunSLSPN.ds[RunSLSPN.data_id].newInstance();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		GraphSPN spn = GraphSPN.load(Parameter.filename, d);
		
		double LL = 0, LLsq = 0;
		long tic = System.currentTimeMillis();
		for(int inst=0; inst<d.getNumTesting(); inst++){
			double ill = 0;
			
			d.show(inst, Partition.Testing);
			ill = spn.upwardPass();
			
			System.out.println(ill);
			LL += ill;
			LLsq += ill*ill;
		}
		long toc = System.currentTimeMillis();
		LL /= d.getNumTesting();
		LLsq /= d.getNumTesting();
		
		System.out.println("avg = "+LL+" +/- "+Math.sqrt(LLsq - LL*LL));
		System.out.println("Total time: "+(1.0*(toc-tic)/1000)+"s");
		// avg = -21.734815 +/- 0.363440
		// Total time: 424.504467s

	}

}
