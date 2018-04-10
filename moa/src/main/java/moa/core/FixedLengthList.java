package moa.core;

import java.util.ArrayList;

public class FixedLengthList<E> extends ArrayList<E>
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 5763493643390042080L;
	
	private int maxSize;
	
	public FixedLengthList(int m)
	{
		maxSize = m;
	}
	
	public boolean add(E entry)
	{
		super.add(entry);
		
		if(this.size() > this.maxSize)
		{
			super.removeRange(0, this.size() - maxSize);
		}
		
		return true;
	}
	
	public E getOldestEntry()
	{
		return super.get(0);
	}
	
	public E getYoungestEntry()
	{
		return super.get(this.size());
	}

}
